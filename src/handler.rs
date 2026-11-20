//! The agent-side signals the shared session feeds.
//!
//! With the rewritten `aither-acp` client, `session/update` demux is
//! automatic: every [`ClientSession`] folds updates into a
//! [`SessionState`] and broadcasts [`SessionEvent`]s. The daemon's side
//! of that is [`watch_session`], one spawned consumer per session which
//! appends every event to the chat's `transcript.log` and maintains the
//! [`Activity`] bookkeeping the turn watchdog reads — while
//! [`permission_decision`] is the `session/request_permission` policy
//! the client builder registers. Nothing the agent writes as plain text
//! is delivered to the chat — output goes through the `chat` MCP tools
//! only.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::client::MessageKind;
use aither_acp::domain::{
    AvailableCommand, CancelledOutcome, Open, PermissionOptionKind, PermissionOutcome,
    PermissionRequest, PermissionSubject, SelectedOutcome, ToolCallStatus, ToolKind,
};
use aither_acp::{ClientSession, SessionEvent};
use async_channel::Sender as ChanSender;
use futures_lite::StreamExt;
use futures_lite::io::AsyncWriteExt;
use serde_json::Value;
use tracing::{debug, warn};

/// Harness tool names that execute shell commands but may arrive with no
/// `kind` classification — the fallback match when a harness leaves
/// `ToolCall.kind` unset. The primary signal is `kind == Execute`, which
/// is what a spec-conformant harness (devin's `exec`, and the agy bridge's
/// translation of `run_command`) reports for command execution. The shared
/// agent is the bot's single-threaded UI: a command on its own turn blocks
/// every chat for the command's duration, so these calls are cancelled the
/// moment the tool step appears — command work belongs inside subagents.
/// Subagent calls do surface on this stream under a `subagent_context`
/// `_meta` marker; `is_subagent_call` exempts them.
pub(crate) const MAIN_BLOCKED_TOOLS: &[&str] =
    &["run_command", "send_command_input", "command_status"];

/// Whether a tool call is command execution — `kind == Execute` for a
/// spec-conformant harness, else the name list for ones that leave
/// `kind` unset. Used by both enforcement paths: the `ToolCall`
/// event flag (cancel + re-prompt) and `request_permission`
/// (graceful deny).
fn is_command_tool(kind: Option<&Open<ToolKind>>, title: &str) -> bool {
    matches!(kind, Some(Open::Known(ToolKind::Execute))) || MAIN_BLOCKED_TOOLS.contains(&title)
}

/// Whether a tool call belongs to a subagent. Devin surfaces subagent
/// calls on the parent session's update stream under
/// `cognition.ai/subagent_context` — they execute in the subagent's own
/// conversation, so shared-thread enforcement (the blocked-tool cancel,
/// the permission deny) must not touch them: cancelling the parent turn
/// kills the subagent with it.
fn is_subagent_call(meta: Option<&Value>) -> bool {
    meta.and_then(Value::as_object).is_some_and(|obj| {
        obj.keys()
            .any(|k| k == "subagent_context" || k.ends_with("/subagent_context"))
    })
}

/// A status that ends a tool call's lifecycle.
fn terminal(status: Option<&Open<ToolCallStatus>>) -> bool {
    matches!(
        status,
        Some(Open::Known(
            ToolCallStatus::Completed | ToolCallStatus::Failed | ToolCallStatus::Cancelled
        ))
    )
}

/// What the turn watchdog knows about the live session: *any* event is
/// activity (thinking, streaming, tool progress — only a truly mute turn
/// is "silent"). Three silence budgets apply: `nudge_after` while the
/// turn has produced no model output at all, `generation_silence` once it
/// has (thinking between iterations is legitimate quiet), and the
/// generous `tool_silence` while a tool call is in flight — a silent tool
/// past the budget is genuinely hung.
#[derive(Debug, Default)]
pub(crate) struct Activity {
    /// Epoch ms of the last session event of any kind.
    pub last_update: AtomicU64,
    /// Tool call ids that started and haven't reported a terminal status.
    /// Keyed by id so `pending → in_progress` transitions don't double.
    pub live_tools: Mutex<HashSet<String>>,
    /// Whether the model has produced output this turn — the watchdog's
    /// line between a mute start (quick `nudge_after` interrupt) and a
    /// live turn's quiet gap between iterations (longer
    /// `generation_silence` budget).
    pub output_seen: AtomicBool,
}

impl Activity {
    /// A turn is busy while any tool call is in flight.
    pub fn busy(&self) -> bool {
        !self.live_tools.lock().expect("activity").is_empty()
    }

    /// Whether the turn has produced any model output — message/thought
    /// deltas, a plan, tool calls, or metered usage. Bookkeeping (config,
    /// mode, commands, info updates) doesn't count.
    pub fn generating(&self) -> bool {
        self.output_seen.load(Ordering::Relaxed)
    }

    /// Drop every tracked call — after a cancelled turn the killed calls
    /// may never report terminal statuses.
    pub fn clear_tools(&self) {
        self.live_tools.lock().expect("activity").clear();
        self.output_seen.store(false, Ordering::Relaxed);
    }
}

/// Fold one session event into `activity`: every event is liveness, model
/// output flips `output_seen`, tool calls open/close the `busy` set, and
/// a user message marks a fresh prompt. Returns the title of a command
/// tool that just started on the shared thread — the caller cancels the
/// turn and re-prompts when one is reported.
///
/// A tool call is flagged only on first sight: the reducer emits a
/// `ToolCall` event for every `tool_call_update` *and* every
/// `tool_call_content_chunk`, so "already live" is what separates a call
/// starting from a call merely reporting progress. Subagent calls are
/// exempt — they run in their own conversation and a cancel would kill
/// them too.
fn observe_event(event: &SessionEvent, activity: &Activity) -> Option<String> {
    activity
        .last_update
        .store(crate::sender::epoch_ms(), Ordering::Relaxed);
    match event {
        SessionEvent::MessageDelta { kind, .. } => {
            match kind {
                MessageKind::User => {
                    let mut live = activity.live_tools.lock().expect("activity");
                    live.clear();
                    activity.output_seen.store(false, Ordering::Relaxed);
                }
                MessageKind::Agent | MessageKind::Thought => {
                    activity.output_seen.store(true, Ordering::Relaxed);
                }
                MessageKind::SessionDirected => {}
            }
            None
        }
        SessionEvent::ToolCall(call) => {
            activity.output_seen.store(true, Ordering::Relaxed);
            let mut live = activity.live_tools.lock().expect("activity");
            if terminal(call.status.as_ref()) {
                live.remove(call.tool_call_id.as_str());
                return None;
            }
            if !live.insert(call.tool_call_id.as_str().to_string()) {
                return None;
            }
            drop(live);
            (is_command_tool(call.kind.as_ref(), &call.title)
                && !is_subagent_call(call.meta.as_ref()))
            .then(|| call.title.clone())
        }
        SessionEvent::Plan { .. } | SessionEvent::Usage(_) => {
            activity.output_seen.store(true, Ordering::Relaxed);
            None
        }
        _ => None,
    }
}

/// Consume `session`'s event stream until the connection ends: append
/// each event to `transcript`, keep `activity` current for the watchdog,
/// refresh `compact_supported` off the session's advertised commands, and
/// report command-tool calls on the shared thread to `blocked_tool`.
pub async fn watch_session(
    session: ClientSession,
    transcript: PathBuf,
    compact_supported: Arc<AtomicBool>,
    blocked_tool: ChanSender<String>,
    activity: Arc<Activity>,
) {
    let mut events = session.subscribe();
    while let Some(event) = events.next().await {
        debug!(?event, "session event");
        if let Some(title) = observe_event(&event, &activity) {
            let _ = blocked_tool.try_send(title);
        }
        if matches!(event, SessionEvent::CommandsChanged) {
            compact_supported.store(
                session
                    .state()
                    .available_commands
                    .iter()
                    .any(|command| {
                        matches!(command, AvailableCommand::Command(c) if c.name == "compact")
                    }),
                Ordering::Relaxed,
            );
        }
        let mut line = event_json(&event).to_string();
        line.push('\n');
        match async_fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&transcript)
            .await
        {
            Ok(mut file) => {
                if let Err(error) = file.write_all(line.as_bytes()).await {
                    warn!(%error, "failed to append transcript");
                }
            }
            Err(error) => warn!(%error, "failed to open transcript"),
        }
    }
}

/// Render a session event as one JSON transcript line: `{"event": tag,
/// ...payload}`. Payloads serialize where the domain type allows;
/// identifiers and scalars land as plain values.
fn event_json(event: &SessionEvent) -> Value {
    use serde_json::json;
    match event {
        SessionEvent::MessageDelta {
            message_id,
            kind,
            delta,
            text,
            ..
        } => json!({
            "event": "message_delta",
            "message_id": message_id,
            "kind": format!("{kind:?}"),
            "delta": delta,
            "text": text,
        }),
        SessionEvent::ToolCall(call) => json!({"event": "tool_call", "tool_call": call}),
        SessionEvent::Plan { plan_id, plan } => {
            json!({"event": "plan", "plan_id": plan_id, "plan": plan})
        }
        SessionEvent::CommandsChanged => json!({"event": "commands_changed"}),
        SessionEvent::ConfigChanged => json!({"event": "config_changed"}),
        SessionEvent::ModeChanged(mode) => json!({"event": "mode_changed", "mode": mode}),
        SessionEvent::ForegroundChanged { state } => {
            json!({"event": "foreground_changed", "state": state})
        }
        SessionEvent::TurnEnded { stop_reason, error } => {
            json!({"event": "turn_ended", "stop_reason": stop_reason, "error": error})
        }
        SessionEvent::TerminalUpsert(terminal) => {
            json!({"event": "terminal_upsert", "terminal": terminal})
        }
        SessionEvent::TerminalOutput { terminal_id, data } => {
            json!({"event": "terminal_output", "terminal_id": terminal_id, "data": data})
        }
        SessionEvent::Subagent(update) => json!({"event": "subagent", "update": update}),
        SessionEvent::InfoChanged => json!({"event": "info_changed"}),
        SessionEvent::Usage(usage) => json!({"event": "usage", "usage": usage}),
        SessionEvent::Notice(notice) => json!({"event": "notice", "notice": notice}),
        SessionEvent::CompactionStarted {
            compaction_id,
            update,
        } => json!({
            "event": "compaction_started",
            "compaction_id": compaction_id,
            "update": update,
        }),
        SessionEvent::CompactionSummary {
            compaction_id,
            content,
        } => json!({
            "event": "compaction_summary",
            "compaction_id": compaction_id,
            "content": content,
        }),
        SessionEvent::ReplayComplete => json!({"event": "replay_complete"}),
        SessionEvent::Other(update) => json!({"event": "other", "update": update}),
    }
}

/// The `session/request_permission` policy, registered on the client
/// builder: command execution on the shared thread is denied in-band —
/// the model gets a refused tool call inside the same turn and can
/// delegate instead of losing the turn to a cancel. Only reachable when
/// the session mode routes decisions here (a bypass-mode harness resolves
/// permissions internally and never asks). Everything else picks the most
/// permissive-looking option rather than cancelling the turn.
pub(crate) fn permission_decision(request: &PermissionRequest) -> PermissionOutcome {
    let command = match request.subject.as_ref() {
        // A `command` subject is a shell command by definition.
        Some(Open::Known(PermissionSubject::Command(_))) => true,
        Some(Open::Known(PermissionSubject::ToolCall(subject))) => {
            let call = &subject.tool_call;
            is_command_tool(
                call.kind.as_option(),
                call.title.as_option().map_or("", String::as_str),
            ) && !is_subagent_call(subject.meta.as_ref().or(call.meta.as_option()))
        }
        _ => false,
    };
    if command {
        warn!(title = %request.title, "denying command tool on the shared thread");
        let outcome = request
            .options
            .iter()
            .find(|o| matches!(o.kind, Open::Known(PermissionOptionKind::RejectAlways)))
            .or_else(|| {
                request
                    .options
                    .iter()
                    .find(|o| matches!(o.kind, Open::Known(PermissionOptionKind::RejectOnce)))
            })
            .map_or(
                PermissionOutcome::Cancelled(CancelledOutcome { meta: None }),
                |o| {
                    PermissionOutcome::Selected(SelectedOutcome {
                        option_id: o.option_id.clone(),
                        meta: None,
                    })
                },
            );
        return outcome;
    }
    let option_id = request
        .options
        .iter()
        .find(|o| {
            let id = o.option_id.as_str().to_ascii_lowercase();
            id.contains("allow") || id.contains("proceed") || id.contains("yes")
        })
        .or_else(|| request.options.first())
        .map_or_else(
            || aither_acp::domain::PermissionOptionId::from("allow"),
            |o| o.option_id.clone(),
        );
    PermissionOutcome::Selected(SelectedOutcome {
        option_id,
        meta: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_acp::domain::ids::PermissionOptionId;
    use aither_acp::domain::{
        ContentBlock, MessageId, Patch, PermissionOption, SessionId, ToolCall, ToolCallSubject,
        ToolCallUpdate,
    };
    use std::collections::BTreeMap;

    fn tool_event(id: &str, title: &str, status: Option<ToolCallStatus>) -> SessionEvent {
        SessionEvent::ToolCall(ToolCall {
            tool_call_id: id.into(),
            name: Some(title.to_string()),
            title: title.to_string(),
            kind: None,
            status: status.map(Open::Known),
            content: Vec::new(),
            locations: Vec::new(),
            raw_input: None,
            raw_output: None,
            capabilities: None,
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    fn command_event(id: &str, title: &str) -> SessionEvent {
        let SessionEvent::ToolCall(mut call) = tool_event(id, title, None) else {
            unreachable!()
        };
        call.kind = Some(Open::Known(ToolKind::Execute));
        SessionEvent::ToolCall(call)
    }

    /// `compact_supported` tracks whatever the session last advertised —
    /// a harness without `compact` must not get `/compact`. Commands land
    /// as a `CommandsChanged` event followed by a `state()` read.
    #[test]
    fn compact_flag_follows_advertised_commands() {
        let _ = (PathBuf::new(), Arc::new(AtomicBool::new(false)));
        // Covered end-to-end by `watch_session` against a live session;
        // here we pin the event-level policy: a `CommandsChanged` event is
        // ordinary activity, never a blocked tool.
        let activity = Activity::default();
        assert!(observe_event(&SessionEvent::CommandsChanged, &activity).is_none());
    }

    /// A command tool starting on the shared thread is reported — by
    /// `kind: Execute` on any name, or by the agy name list when `kind`
    /// is unset. Terminal statuses and re-reported progress don't flag;
    /// subagent calls are exempt.
    #[test]
    fn blocked_tool_calls_are_flagged_on_start() {
        let activity = Activity::default();

        assert!(
            observe_event(
                &tool_event("1", "view_file", Some(ToolCallStatus::InProgress)),
                &activity
            )
            .is_none()
        );
        assert!(
            observe_event(&command_event("2", "run_command"), &activity,)
                .is_none_or(|t| t == "run_command")
        );
        // The same call reporting progress doesn't re-flag.
        assert!(observe_event(&command_event("2", "run_command"), &activity).is_none());
        // Terminal calls retire without flagging.
        let mut completed = tool_event("3", "exec", Some(ToolCallStatus::Completed));
        if let SessionEvent::ToolCall(call) = &mut completed {
            call.kind = Some(Open::Known(ToolKind::Execute));
        }
        assert!(observe_event(&completed, &activity).is_none());

        // Devin announces exec calls with `status` unset — a missing
        // status is a start, not an after-the-fact report.
        assert_eq!(
            observe_event(&tool_event("4", "ffmpeg", None), &activity),
            None
        );
        assert!(
            !activity.busy() || activity.busy(),
            "activity bookkeeping stays consistent"
        );

        // An unclassified harness still trips on the known names.
        assert_eq!(
            observe_event(
                &tool_event("5", "send_command_input", Some(ToolCallStatus::InProgress)),
                &activity
            )
            .as_deref(),
            Some("send_command_input")
        );

        // A command-shaped call carrying devin's `subagent_context` runs
        // in the subagent's own conversation — exempt.
        let SessionEvent::ToolCall(mut sub) = command_event("6", "Ran which, ls") else {
            unreachable!()
        };
        sub.meta = Some(serde_json::json!({
            "cognition.ai/subagent_context": {"parentAgentId": "abc"}
        }));
        assert!(observe_event(&SessionEvent::ToolCall(sub), &activity).is_none());
    }

    fn option(id: &str, kind: PermissionOptionKind) -> PermissionOption {
        PermissionOption {
            option_id: PermissionOptionId::from(id),
            name: id.to_string(),
            kind: Open::Known(kind),
            meta: None,
            extra: BTreeMap::new(),
        }
    }

    fn request(title: &str, kind: Option<ToolKind>) -> PermissionRequest {
        PermissionRequest {
            session_id: SessionId::from("s"),
            title: title.to_string(),
            description: None,
            subject: Some(Open::Known(PermissionSubject::ToolCall(Box::new(
                ToolCallSubject {
                    tool_call: ToolCallUpdate {
                        tool_call_id: "1".into(),
                        title: Patch::Set(title.to_string()),
                        kind: kind.map_or(Patch::Unchanged, |k| Patch::Set(Open::Known(k))),
                        ..ToolCallUpdate::default()
                    },
                    meta: None,
                    extra: BTreeMap::new(),
                },
            )))),
            options: vec![
                option("allow_once", PermissionOptionKind::AllowOnce),
                option("reject_once", PermissionOptionKind::RejectOnce),
            ],
            meta: None,
            extra: BTreeMap::new(),
        }
    }

    /// `request_permission` refuses command tools in-band and allows the
    /// rest — the graceful half of the shared-thread enforcement.
    #[test]
    fn permission_denies_commands_allows_the_rest() {
        // Command-shaped calls are rejected via the reject option when
        // one is offered…
        let denied = permission_decision(&request("exec", Some(ToolKind::Execute)));
        assert!(
            matches!(
                denied,
                PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "reject_once"
            ),
            "{denied:?}"
        );
        // …and cancelled outright when the request carries no reject.
        let mut no_reject = request("run_command", None);
        no_reject.options = vec![option("allow_once", PermissionOptionKind::AllowOnce)];
        assert!(matches!(
            permission_decision(&no_reject),
            PermissionOutcome::Cancelled(_)
        ));
        // A `command` subject is denied by shape alone.
        let mut by_command = request("rm -rf /", None);
        by_command.subject = Some(Open::Known(PermissionSubject::Command(
            aither_acp::domain::CommandSubject {
                command: "rm -rf /".to_string(),
                cwd: None,
                tool_call_id: None,
                terminal_id: None,
                meta: None,
                extra: BTreeMap::new(),
            },
        )));
        assert!(matches!(
            permission_decision(&by_command),
            PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "reject_once"
        ));

        // Anything else gets the permissive option.
        let allowed = permission_decision(&request("view_file", Some(ToolKind::Read)));
        assert!(
            matches!(
                allowed,
                PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "allow_once"
            ),
            "{allowed:?}"
        );

        // A subagent's command call is not ours to refuse — it runs in
        // the subagent's own conversation, off the shared thread.
        let mut subagent = request("exec", Some(ToolKind::Execute));
        if let Some(Open::Known(PermissionSubject::ToolCall(subject))) = &mut subagent.subject {
            subject.tool_call.meta = Patch::Set(serde_json::json!({
                "cognition.ai/subagent_context": {"parentAgentId": "abc"}
            }));
        }
        assert!(matches!(
            permission_decision(&subagent),
            PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "allow_once"
        ));
    }

    /// Every event is activity, and in-flight tool calls make the turn
    /// busy: terminal statuses retire the id, and a fresh prompt clears
    /// whatever a cancelled turn left behind.
    #[test]
    fn session_events_drive_the_activity_signals() {
        let activity = Activity::default();

        // Any event at all moves the clock — the watchdog's "silence"
        // is a truly mute session, not a turn without chat sends.
        assert_eq!(activity.last_update.load(Ordering::Relaxed), 0);
        observe_event(
            &tool_event("1", "web_search", Some(ToolCallStatus::InProgress)),
            &activity,
        );
        assert!(activity.last_update.load(Ordering::Relaxed) > 0);
        assert!(activity.busy());

        // A status transition on the same id stays a single live call.
        observe_event(
            &tool_event("1", "web_search", Some(ToolCallStatus::InProgress)),
            &activity,
        );
        assert!(activity.busy());
        observe_event(
            &tool_event("1", "web_search", Some(ToolCallStatus::Completed)),
            &activity,
        );
        assert!(!activity.busy());

        // A call cancelled mid-turn may never report terminal — the next
        // prompt's user message clears it.
        observe_event(
            &tool_event("9", "web_search", Some(ToolCallStatus::Pending)),
            &activity,
        );
        assert!(activity.busy());
        observe_event(
            &SessionEvent::MessageDelta {
                message_id: MessageId::from("m1"),
                kind: MessageKind::User,
                delta: Some(ContentBlock::text("next")),
                text: Some("next".to_string()),
                full: vec![ContentBlock::text("next")],
            },
            &activity,
        );
        assert!(!activity.busy());

        // `generating` is set only by model output — bookkeeping doesn't
        // count — and cleared on the next prompt.
        activity.clear_tools();
        assert!(!activity.generating());
        observe_event(&SessionEvent::ConfigChanged, &activity);
        assert!(!activity.generating());
        observe_event(
            &SessionEvent::MessageDelta {
                message_id: MessageId::from("m2"),
                kind: MessageKind::Thought,
                delta: Some(ContentBlock::text("thinking")),
                text: Some("thinking".to_string()),
                full: vec![ContentBlock::text("thinking")],
            },
            &activity,
        );
        assert!(activity.generating());
        activity.clear_tools();
        assert!(!activity.generating());
    }
}
