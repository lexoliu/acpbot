//! The ACP client-side handler: what the agent says back to us.
//!
//! Session updates (message chunks, tool calls, plans) are appended to the
//! chat's `transcript.log` and logged via `tracing`. Nothing the agent writes
//! as plain text is delivered to the chat — output goes through the `chat`
//! MCP tools only.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    ClientHandler, RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult,
    SessionNotification, SessionUpdate, ToolCallStatus, ToolKind,
};
use aither_mcp::protocol::JsonRpcError;
use async_channel::Sender as ChanSender;
use futures_lite::io::AsyncWriteExt;
use tracing::{debug, warn};

/// Harness tool names that execute shell commands but may arrive with no
/// `kind` classification — the fallback match when a harness leaves
/// `ToolCall.kind` unset. The primary signal is `kind == Execute`, which
/// is what a spec-conformant harness (devin's `exec`, and the agy bridge's
/// translation of `run_command`) reports for command execution. The shared
/// agent is the bot's single-threaded UI: a command on its own turn blocks
/// every chat for the command's duration, so these calls are cancelled the
/// moment the tool step appears — command work belongs inside subagents,
/// whose tool calls run in their own conversations and never surface here.
pub(crate) const MAIN_BLOCKED_TOOLS: &[&str] =
    &["run_command", "send_command_input", "command_status"];

/// What the turn watchdog knows about the live session: *any* update is
/// activity (thinking, streaming, tool progress — only a truly mute turn
/// is "silent"), and in-flight tool calls make the turn *busy* — it gets
/// the generous `tool_silence` budget instead of `nudge_after`, because a
/// `web_search`/`exec`/subagent working quietly for a minute is normal,
/// while a silent tool past the budget is genuinely hung.
#[derive(Debug, Default)]
pub(crate) struct Activity {
    /// Epoch ms of the last session notification of any kind.
    pub last_update: AtomicU64,
    /// Tool call ids that started and haven't reported a terminal status.
    /// Keyed by id so `pending → in_progress` transitions don't double.
    pub live_tools: Mutex<HashSet<String>>,
}

impl Activity {
    /// A turn is busy while any tool call is in flight.
    pub fn busy(&self) -> bool {
        !self.live_tools.lock().expect("activity").is_empty()
    }

    /// Drop every tracked call — after a cancelled turn the killed calls
    /// may never report terminal statuses.
    pub fn clear_tools(&self) {
        self.live_tools.lock().expect("activity").clear();
    }

    /// A status that ends a tool call's lifecycle.
    fn terminal(status: Option<ToolCallStatus>) -> bool {
        matches!(
            status,
            Some(ToolCallStatus::Completed | ToolCallStatus::Failed)
        )
    }

    /// Record one tool-call status: terminal retires the id, anything
    /// else marks it live (a `pending` → `in_progress` transition just
    /// re-inserts the same id).
    fn track(live: &mut HashSet<String>, id: &str, status: Option<ToolCallStatus>) {
        if Self::terminal(status) {
            live.remove(id);
        } else {
            live.insert(id.to_string());
        }
    }
}

/// Handles agent-to-client traffic for the shared session.
#[derive(Debug)]
pub struct BotClientHandler {
    /// File the session's updates are appended to, one JSON line each.
    transcript: PathBuf,
    /// Set when the session's `available_commands_update` advertises a
    /// `compact` command — the actor reads it to decide whether an idle
    /// compaction is safe to send (`/compact` is devin-specific; other
    /// harnesses would treat it as ordinary text).
    compact_supported: Arc<AtomicBool>,
    /// Reports a blocked tool the moment its call starts; the actor
    /// cancels the turn and re-prompts with a delegation nudge.
    blocked_tool: ChanSender<String>,
    /// Activity signals the watchdog reads — written on every update.
    activity: Arc<Activity>,
}

impl BotClientHandler {
    /// A handler appending to `transcript`, publishing command support
    /// flags onto `compact_supported`, reporting calls to
    /// `MAIN_BLOCKED_TOOLS` on `blocked_tool`, and recording session
    /// activity into `activity`.
    pub fn new(
        transcript: PathBuf,
        compact_supported: Arc<AtomicBool>,
        blocked_tool: ChanSender<String>,
        activity: Arc<Activity>,
    ) -> Self {
        Self {
            transcript,
            compact_supported,
            blocked_tool,
            activity,
        }
    }
}

impl ClientHandler for BotClientHandler {
    async fn session_update(&self, notification: SessionNotification) {
        let update = &notification.update;
        debug!(?update, "session update");
        self.activity
            .last_update
            .store(crate::sender::epoch_ms(), Ordering::Relaxed);

        // Tool-call lifecycle for the busy flag. A `user_message_chunk`
        // means a fresh prompt: whatever ran before is dead, whether or
        // not the harness bothered to say so.
        {
            let mut live = self.activity.live_tools.lock().expect("activity");
            match update {
                SessionUpdate::ToolCall(call) => {
                    Activity::track(&mut live, &call.tool_call_id, call.status);
                }
                SessionUpdate::ToolCallUpdate(call) => {
                    Activity::track(&mut live, &call.tool_call_id, call.status);
                }
                SessionUpdate::UserMessageChunk(_) => live.clear(),
                _ => {}
            }
        }

        if let SessionUpdate::AvailableCommandsUpdate(commands) = update {
            self.compact_supported.store(
                commands
                    .available_commands
                    .iter()
                    .any(|command| command.name == "compact"),
                Ordering::Relaxed,
            );
        }

        // A command tool starting on the shared thread: flag it so the
        // turn loop can cancel before the command's duration becomes
        // every chat's wait. `kind == Execute` is the harness-agnostic
        // signal; the name list catches harnesses that leave `kind` unset.
        // Completed-status updates don't re-report.
        if let SessionUpdate::ToolCall(call) = update
            && call.status == Some(ToolCallStatus::InProgress)
            && (call.kind == Some(ToolKind::Execute)
                || MAIN_BLOCKED_TOOLS.contains(&call.title.as_str()))
        {
            let _ = self.blocked_tool.try_send(call.title.clone());
        }

        let Ok(mut line) = serde_json::to_string(update) else {
            return;
        };
        line.push('\n');
        match async_fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.transcript)
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

    async fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> Result<RequestPermissionResult, JsonRpcError> {
        // Sessions run in bypass mode, so this is a last-resort path: pick the
        // most permissive-looking option rather than cancelling the turn.
        let option_id = params
            .options
            .iter()
            .find(|o| {
                let id = o.option_id.to_ascii_lowercase();
                id.contains("allow") || id.contains("proceed") || id.contains("yes")
            })
            .or_else(|| params.options.first())
            .map_or_else(|| "allow".to_string(), |o| o.option_id.clone());
        Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Selected { option_id },
            meta: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_acp::{AvailableCommand, AvailableCommandsUpdate, SessionUpdate};
    use futures_lite::future::block_on;

    fn notification(names: &[&str]) -> SessionNotification {
        SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate {
                available_commands: names
                    .iter()
                    .map(|name| AvailableCommand {
                        name: (*name).to_string(),
                        description: String::new(),
                        input: None,
                        meta: None,
                    })
                    .collect(),
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        }
    }

    /// The `compact_supported` flag tracks whatever the session last
    /// advertised — a harness without `compact` must not get `/compact`.
    #[test]
    fn compact_flag_follows_advertised_commands() {
        let flag = Arc::new(AtomicBool::new(false));
        let (_tx, rx) = async_channel::unbounded::<String>();
        let handler = BotClientHandler::new(
            PathBuf::from("/tmp/acpbot-test-transcript.log"),
            flag.clone(),
            _tx,
            Arc::new(Activity::default()),
        );
        assert!(rx.is_empty());

        block_on(handler.session_update(notification(&["login", "ask"])));
        assert!(!flag.load(Ordering::Relaxed));

        block_on(handler.session_update(notification(&["login", "compact"])));
        assert!(flag.load(Ordering::Relaxed));
    }

    /// A command tool starting on the shared thread is reported on the
    /// blocked channel — by `kind: Execute` on any name, or by the agy
    /// name list when `kind` is unset. Completion updates and unrelated
    /// tools aren't.
    #[test]
    fn blocked_tool_calls_are_flagged_on_start() {
        use aither_acp::ToolCall;
        let (tx, rx) = async_channel::unbounded::<String>();
        let handler = BotClientHandler::new(
            PathBuf::from("/tmp/acpbot-test-transcript.log"),
            Arc::new(AtomicBool::new(false)),
            tx,
            Arc::new(Activity::default()),
        );
        let call = |title: &str, kind: Option<ToolKind>, status| SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::ToolCall(ToolCall {
                tool_call_id: "1".to_string(),
                title: title.to_string(),
                kind,
                status: Some(status),
                content: Vec::new(),
                locations: Vec::new(),
                raw_input: None,
                raw_output: None,
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        };

        block_on(handler.session_update(call(
            "view_file",
            Some(ToolKind::Read),
            ToolCallStatus::InProgress,
        )));
        block_on(handler.session_update(call(
            "run_command",
            Some(ToolKind::Execute),
            ToolCallStatus::Completed,
        )));
        assert!(rx.is_empty());

        // A spec-conformant harness (devin's `exec`) reports the kind —
        // any name with `Execute` is a command.
        block_on(handler.session_update(call(
            "exec",
            Some(ToolKind::Execute),
            ToolCallStatus::InProgress,
        )));
        assert_eq!(rx.try_recv().as_deref(), Ok("exec"));

        // An unclassified harness still trips on the known names.
        block_on(handler.session_update(call(
            "send_command_input",
            None,
            ToolCallStatus::InProgress,
        )));
        assert_eq!(rx.try_recv().as_deref(), Ok("send_command_input"));
    }

    /// Every update is activity, and in-flight tool calls make the turn
    /// busy: terminal statuses retire the id, and a fresh prompt clears
    /// whatever a cancelled turn left behind.
    #[test]
    fn session_updates_drive_the_activity_signals() {
        use aither_acp::{ContentBlock, ContentChunk, TextContent, ToolCall, ToolCallUpdate};
        let activity = Arc::new(Activity::default());
        let (tx, _rx) = async_channel::unbounded::<String>();
        let handler = BotClientHandler::new(
            PathBuf::from("/tmp/acpbot-test-transcript.log"),
            Arc::new(AtomicBool::new(false)),
            tx,
            activity.clone(),
        );
        let tool = |id: &str, status| SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::ToolCall(ToolCall {
                tool_call_id: id.to_string(),
                title: "web_search".to_string(),
                kind: None,
                status: Some(status),
                content: Vec::new(),
                locations: Vec::new(),
                raw_input: None,
                raw_output: None,
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        };
        let tool_update = |id: &str, status| SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                tool_call_id: id.to_string(),
                status: Some(status),
                title: None,
                kind: None,
                content: None,
                locations: None,
                raw_input: None,
                raw_output: None,
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        };

        // Any update at all moves the clock — the watchdog's "silence"
        // is a truly mute session, not a turn without chat sends.
        assert_eq!(activity.last_update.load(Ordering::Relaxed), 0);
        block_on(handler.session_update(tool("1", ToolCallStatus::InProgress)));
        assert!(activity.last_update.load(Ordering::Relaxed) > 0);
        assert!(activity.busy());

        // A status transition on the same id stays a single live call.
        block_on(handler.session_update(tool("1", ToolCallStatus::InProgress)));
        assert!(activity.busy());
        block_on(handler.session_update(tool_update("1", ToolCallStatus::Completed)));
        assert!(!activity.busy());

        // A call cancelled mid-turn may never report terminal — the next
        // prompt's user chunk clears it.
        block_on(handler.session_update(tool("9", ToolCallStatus::Pending)));
        assert!(activity.busy());
        block_on(handler.session_update(SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::UserMessageChunk(ContentChunk {
                content: ContentBlock::Text(TextContent {
                    text: "next".to_string(),
                    annotations: None,
                    meta: None,
                }),
                message_id: None,
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        }));
        assert!(!activity.busy());
    }
}
