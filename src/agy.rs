//! `acpbot agy-bridge` — an ACP server over stdio that fronts the
//! Antigravity `agy` CLI, so `[agent] command = "acpbot"` with
//! `args = ["agy-bridge"]` swaps the harness without touching the daemon.
//!
//! `agy` is not an ACP server: its headless mode is one `stream-json`
//! process per turn — stdin is read to EOF, one `user` event is run, and
//! `init`/`step_update`/`result` NDJSON lines stream out. Conversation
//! state lives server-side under an agy-assigned `conversation_id`, which
//! `--conversation <id>` resumes. So the bridge spawns `agy` per
//! `session/prompt`, threads the conversation id across turns, and
//! persists `acp session id → conversation id` in the session's cwd so
//! `session/load`/`session/resume` keep working across bridge restarts.

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use aither_acp::domain::ids::{ConfigId, ToolCallId};
use aither_acp::domain::{
    AgentCapabilities, ConfigOption, ConfigValue, ContentBlock, Implementation, Open, Patch,
    SessionCaps, SessionId, SessionInfo, SessionListParams, SessionPage, SessionSetupParams,
    SessionUpdate, StopReason, ToolCallStatus, ToolCallUpdate, ToolKind,
};
use aither_acp::rpc::RpcError;
use aither_acp::server::{
    MessageStream, PendingPrompt, PromptError, ResumeRequest, Server, ServerError, Session,
    SessionContext, SessionHost, ToolCallHandle, Turn,
};
use futures_lite::future::or;
use tracing::{debug, warn};

use crate::error::BridgeError;

/// The agy executable — overridable for testing or non-PATH installs.
fn agy_bin() -> String {
    std::env::var("AGY_BIN").unwrap_or_else(|_| "agy".to_string())
}

/// Where agy keeps its global MCP server registry.
fn agy_mcp_config() -> PathBuf {
    dirs::home_dir()
        .expect("home dir")
        .join(".gemini/config/mcp_config.json")
}

/// The acp session → agy conversation map inside a session's cwd.
fn conversation_map_path(cwd: &Path) -> PathBuf {
    cwd.join(".agy-bridge.json")
}

/// The bridge's host: opens sessions for `session/new`, rebinds them for
/// `session/load`/`session/resume`, and tracks the live set for
/// `session/list`.
#[derive(Default)]
struct AgyHost {
    /// `session id → cwd` for every live session — `close`/`delete` remove.
    live: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

/// One ACP session backed by an agy conversation.
struct AgySession {
    /// The ACP id the server assigned — keys the conversation map.
    id: SessionId,
    /// The session's working directory — agy inherits it as its cwd.
    cwd: PathBuf,
    /// `--model` override from a `model` config set.
    model: Mutex<Option<String>>,
    /// `--mode` override from a mode set (accept-edits/plan only).
    mode: Mutex<Option<String>>,
    /// The agy conversation backing this session — known once the first
    /// turn's `init` event reports it.
    conversation: Mutex<Option<String>>,
    /// The running turn's agy child, so `session/close` — and the turn's
    /// own `cancelled` — can kill it while the prompt is in flight.
    running: Arc<Mutex<Option<Child>>>,
    /// Back-reference into the host's live map so `close` unregisters.
    live: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

/// Run the bridge: serve ACP on stdio until the transport ends.
pub fn run() -> Result<(), BridgeError> {
    futures_lite::future::block_on(async {
        let (_peer, serve) = Server::new(
            AgyHost::default(),
            Implementation::new("acpbot agy-bridge", env!("CARGO_PKG_VERSION")),
        )
        .serve_stdio();
        serve.await;
    });
    Ok(())
}

impl AgyHost {
    /// Register a session and build it.
    fn register(
        &self,
        session_id: &SessionId,
        cwd: PathBuf,
        conversation: Option<String>,
    ) -> AgySession {
        mirror_mcp_servers(&cwd);
        self.live
            .lock()
            .expect("live")
            .insert(session_id.clone(), cwd.clone());
        AgySession {
            id: session_id.clone(),
            cwd,
            model: Mutex::new(None),
            mode: Mutex::new(None),
            conversation: Mutex::new(conversation),
            running: Arc::new(Mutex::new(None)),
            live: self.live.clone(),
        }
    }
}

impl SessionHost for AgyHost {
    type Session = AgySession;

    /// The v1 capability surface the daemon's client probes: text-only
    /// prompts, `loadSession`, and resume/close/list under
    /// `sessionCapabilities` — encoded through the caps `extra` keys the
    /// v1 codec reads.
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities {
            session: Some(SessionCaps {
                prompt: Some(aither_acp::domain::PromptCaps::default()),
                delete: Some(aither_acp::domain::EmptyObject),
                extra: BTreeMap::from([
                    ("loadSession".to_string(), serde_json::json!(true)),
                    ("resume".to_string(), serde_json::json!({})),
                    ("close".to_string(), serde_json::json!({})),
                    ("list".to_string(), serde_json::json!({})),
                ]),
                ..SessionCaps::default()
            }),
            ..AgentCapabilities::default()
        }
    }

    /// `session/new`: mirror the chat MCP endpoint into agy's global
    /// registry and open a session; agy itself spawns lazily at the first
    /// prompt so `set_mode`/`set_config_option` land before it does.
    async fn new_session(
        &self,
        params: SessionSetupParams,
        cx: &SessionContext,
    ) -> Result<AgySession, ServerError> {
        Ok(self.register(cx.session_id(), params.cwd, None))
    }

    /// `session/resume` (and v1 `session/load`/`session/fork` behind it):
    /// rebind an acp session id to its recorded agy conversation so a
    /// recovery handoff can still ask the old session for its summary.
    async fn resume_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> Result<AgySession, ServerError> {
        let cwd = req.params.setup.cwd.clone();
        let map = conversation_map(&cwd);
        let session_id = req.session_id().clone();
        let Some(conversation) = map.get(session_id.as_str()).cloned() else {
            return Err(ServerError::Rpc(RpcError::invalid_params(format!(
                "unknown session: {session_id}"
            ))));
        };
        let _ = cx;
        Ok(self.register(&session_id, cwd, Some(conversation)))
    }

    /// `session/list`: the live sessions this bridge still serves.
    async fn list_sessions(&self, _params: SessionListParams) -> Result<SessionPage, RpcError> {
        let sessions = self
            .live
            .lock()
            .expect("live")
            .iter()
            .map(|(id, cwd)| SessionInfo {
                session_id: id.clone(),
                cwd: cwd.clone(),
                title: None,
                updated_at: None,
                meta: None,
                extra: BTreeMap::new(),
            })
            .collect();
        Ok(SessionPage {
            sessions,
            ..SessionPage::default()
        })
    }

    /// `session/delete`: drop the session — the conversation map stays so
    /// a later `session/load` may still need the recovery handle.
    async fn delete_session(&self, session_id: &SessionId) -> Result<(), RpcError> {
        self.live.lock().expect("live").remove(session_id);
        Ok(())
    }
}

impl Session for AgySession {
    /// `session/set_config_option` (v1 `session/set_mode` decodes to the
    /// mode-category option): `model` becomes `--model`; a mode becomes
    /// `--mode` when agy knows it — everything else is accepted and
    /// ignored.
    async fn set_config(
        &self,
        id: &ConfigId,
        value: ConfigValue,
        _cx: &SessionContext,
    ) -> Result<Vec<ConfigOption>, RpcError> {
        if let ConfigValue::Id(by_id) = value {
            if id.as_str() == "model" {
                *self.model.lock().expect("model") = Some(by_id.value.into_string());
            } else {
                *self.mode.lock().expect("mode") = Some(by_id.value.into_string());
            }
        }
        Ok(Vec::new())
    }

    /// `session/prompt`: spawn `agy` for this one turn, feed it the prompt
    /// as a single `user` stream-json event, translate its `step_update`
    /// stream into `session/update`s, and answer with the turn's stop
    /// reason when `result` arrives.
    async fn prompt(
        &self,
        pending: PendingPrompt,
        cx: &SessionContext,
    ) -> Result<StopReason, PromptError> {
        let text = prompt_text(pending.content());
        let session_id = cx.session_id().clone();
        let turn = pending
            .accept()
            .await
            .map_err(|e| PromptError::Failed(e.into_inner()))?;
        Ok(self.run_turn(&turn, &session_id, &text).await)
    }

    /// Kills any in-flight agy child and drops the session from the live
    /// map — the conversation map on disk stays for a later load.
    async fn close(&self) {
        if let Some(mut child) = self.running.lock().expect("running").take() {
            let _ = child.kill();
        }
        self.live.lock().expect("live").remove(&self.id);
    }
}

impl AgySession {
    /// One turn of `agy`, streaming updates through `turn`. Returns the
    /// turn's stop reason.
    async fn run_turn(&self, turn: &Turn, session_id: &SessionId, text: &str) -> StopReason {
        let mut args = vec![
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ];
        if let Some(model) = self.model.lock().expect("model").clone() {
            args.extend(["--model".to_string(), model]);
        }
        if let Some(mode) = self.mode.lock().expect("mode").clone()
            && matches!(mode.as_str(), "accept-edits" | "plan")
        {
            args.extend(["--mode".to_string(), mode]);
        }
        if let Some(conversation) = self.conversation.lock().expect("conversation").clone() {
            args.extend(["--conversation".to_string(), conversation]);
        }
        args.extend(["--print".to_string(), String::new()]);

        let mut child = match Command::new(agy_bin())
            .args(&args)
            .current_dir(&self.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                warn!(%error, "spawn agy failed");
                return StopReason::Error;
            }
        };

        // The whole turn is one line; closing stdin is what starts it —
        // agy reads stream-json input to EOF before running the turn.
        if let Some(mut stdin) = child.stdin.take() {
            let line = serde_json::json!({"event":"user","message":{"content": text}});
            if let Err(error) = writeln!(stdin, "{line}") {
                warn!(%error, "write agy stdin failed");
                let _ = child.kill();
                return StopReason::Error;
            }
        }
        drop(child.stdin.take());

        // agy's stdout is a blocking pipe; a reader thread turns it into a
        // channel so the turn loop can race it against the turn's cancel.
        // `None` on the channel marks end-of-stream.
        let (lines_tx, lines_rx) = async_channel::unbounded::<Option<String>>();
        let stdout = child.stdout.take().expect("piped");
        std::thread::spawn(move || {
            use std::io::BufRead as _;
            let mut reader = std::io::BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if lines_tx.send_blocking(Some(line.clone())).is_err() {
                            return;
                        }
                    }
                }
            }
            let _ = lines_tx.send_blocking(None);
        });

        *self.running.lock().expect("running") = Some(child);

        /// Which side of the turn's `or` produced a value.
        enum Step {
            /// One stdout line, or `None` at end-of-stream.
            Line(Option<String>),
            /// `session/cancel` (or a close) hit the turn.
            Cancel,
        }

        let mut tools: HashMap<String, ToolCallHandle> = HashMap::new();
        let mut message: Option<MessageStream> = None;
        let mut thought: Option<MessageStream> = None;
        let mut cancelled = false;
        let stop = loop {
            let step = or(
                async { Step::Line(lines_rx.recv().await.unwrap_or(None)) },
                async {
                    turn.cancelled().await;
                    Step::Cancel
                },
            )
            .await;
            match step {
                Step::Line(Some(line)) => {
                    if let Some(stop) = self
                        .agy_line(
                            turn,
                            session_id,
                            &mut tools,
                            &mut message,
                            &mut thought,
                            &line,
                        )
                        .await
                    {
                        break stop;
                    }
                }
                // stdout closed with no `result`: killed (cancel) or crashed.
                Step::Line(None) => {
                    break if cancelled {
                        StopReason::Cancelled
                    } else {
                        warn!("agy stream ended without a result event");
                        StopReason::Error
                    };
                }
                Step::Cancel => {
                    cancelled = true;
                    if let Some(mut c) = self.running.lock().expect("running").take() {
                        let _ = c.kill();
                    }
                }
            }
        };

        // Reap the child — normally it has already exited after `result`;
        // bound the wait so a lingering process cannot hang the loop.
        if let Some(mut c) = self.running.lock().expect("running").take() {
            for _ in 0..200 {
                match c.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
            if c.try_wait().ok().flatten().is_none() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }

        stop
    }

    /// Translate one agy NDJSON line into `session/update`s. Returns the
    /// turn's stop reason once the `result` event arrives.
    async fn agy_line(
        &self,
        turn: &Turn,
        session_id: &SessionId,
        tools: &mut HashMap<String, ToolCallHandle>,
        message: &mut Option<MessageStream>,
        thought: &mut Option<MessageStream>,
        line: &str,
    ) -> Option<StopReason> {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            debug!("unparsed agy line: {}", &line[..line.len().min(200)]);
            return None;
        };
        match value.get("event").and_then(|e| e.as_str()) {
            Some("init") => {
                // First sight of the conversation id: bind it so later
                // turns — and a later `session/load` after a crash —
                // resume the same conversation.
                if let Some(conversation_id) = value.get("conversation_id").and_then(|c| c.as_str())
                    && self.conversation.lock().expect("conversation").as_deref()
                        != Some(conversation_id)
                {
                    *self.conversation.lock().expect("conversation") =
                        Some(conversation_id.to_string());
                    let mut map = conversation_map(&self.cwd);
                    map.insert(session_id.to_string(), conversation_id.to_string());
                    save_conversation_map(&self.cwd, &map);
                }
                None
            }
            Some("step_update") => {
                let step = value.get("step_update")?;
                self.step_update(turn, tools, message, thought, step).await;
                None
            }
            Some("result") => Some(
                match value.pointer("/result/status").and_then(|s| s.as_str()) {
                    Some("SUCCESS") => StopReason::EndTurn,
                    Some("CANCELLED") => StopReason::Cancelled,
                    status => {
                        warn!(status = ?status, "agy turn ended abnormally");
                        StopReason::Error
                    }
                },
            ),
            _ => None,
        }
    }

    /// One `step_update` payload → tool calls and message chunks on `turn`.
    async fn step_update(
        &self,
        turn: &Turn,
        tools: &mut HashMap<String, ToolCallHandle>,
        message: &mut Option<MessageStream>,
        thought: &mut Option<MessageStream>,
        step: &serde_json::Value,
    ) {
        let Some(step_type) = step.get("step_type").and_then(|t| t.as_str()) else {
            return;
        };
        match step_type {
            "agent_response" => {
                if let Some(text) = step.get("text_delta").and_then(|t| t.as_str()) {
                    let _ = message
                        .get_or_insert_with(|| turn.message())
                        .text(text)
                        .await;
                }
            }
            "agent_thought" | "thinking" => {
                if let Some(text) = step.get("text_delta").and_then(|t| t.as_str()) {
                    let _ = thought
                        .get_or_insert_with(|| turn.thought())
                        .text(text)
                        .await;
                }
            }
            "tool" => {
                let Some(id) = step.get("step_index").and_then(|i| i.as_u64()) else {
                    return;
                };
                let id = id.to_string();
                let name = step
                    .get("tool_name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default();
                // Classify agy's command tools so the client's blocked-tool
                // rule works on `kind` — harness-agnostic — rather than
                // these agy-specific names.
                let kind = crate::handler::MAIN_BLOCKED_TOOLS
                    .contains(&name)
                    .then_some(ToolKind::Execute);
                if step.get("state").and_then(|s| s.as_str()) == Some("ACTIVE") {
                    let mut builder = turn.tool_call(name, name);
                    if let Some(kind) = kind {
                        builder = builder.kind(kind);
                    }
                    if let Some(input) = step.get("tool_info").and_then(|i| i.get("parameters")) {
                        builder = builder.input(input.clone());
                    }
                    match builder.start().await {
                        Ok(handle) => {
                            let _ = handle.in_progress().await;
                            tools.insert(id, handle);
                        }
                        Err(error) => debug!(%error, "tool_call update failed"),
                    }
                } else {
                    let output = step.get("tool_info").and_then(|i| i.get("output")).cloned();
                    match tools.remove(&id) {
                        Some(handle) => {
                            let _ = handle.complete(output).await;
                        }
                        // A terminal step without a seen ACTIVE — emit the
                        // completion raw so the client still retires it.
                        None => {
                            let mut update = ToolCallUpdate::new(ToolCallId::from(id.as_str()));
                            update.status = Patch::Set(Open::Known(ToolCallStatus::Completed));
                            if let Some(output) = output {
                                update.raw_output = Patch::Set(output);
                            }
                            let _ = turn.update(SessionUpdate::ToolCallUpdate(update)).await;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The concatenated text of a prompt's text blocks; image/audio payloads
/// are not sent to agy (its stream-json input is text) — the event JSON
/// already carries their `inbox/` paths for the file tools.
fn prompt_text(prompt: &[ContentBlock]) -> String {
    prompt
        .iter()
        .filter_map(|block| block.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Read a session cwd's `acp session id → agy conversation` map.
fn conversation_map(cwd: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(conversation_map_path(cwd))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Persist the conversation map atomically — a crash mid-write must not
/// strand a recoverable session.
fn save_conversation_map(cwd: &Path, map: &HashMap<String, String>) {
    let tmp = conversation_map_path(cwd).with_extension("tmp");
    if let Ok(text) = serde_json::to_string(map)
        && std::fs::write(&tmp, text).is_ok()
    {
        let _ = std::fs::rename(&tmp, conversation_map_path(cwd));
    }
}

/// Mirror the chat MCP servers acpbot wrote into `.devin/mcp_config.json`
/// into agy's global registry (`~/.gemini/config/mcp_config.json`) — agy
/// has no per-project MCP config, so the bridge registers by name.
fn mirror_mcp_servers(cwd: &Path) {
    let Ok(text) = std::fs::read_to_string(cwd.join(".devin/mcp_config.json")) else {
        return;
    };
    let Ok(local) = serde_json::from_str::<serde_json::Value>(&text) else {
        return;
    };
    let Some(servers) = local.get("mcpServers").and_then(|s| s.as_object()) else {
        return;
    };

    let global_path = agy_mcp_config();
    let mut global: serde_json::Value = std::fs::read_to_string(&global_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({"mcpServers":{}}));
    let Some(registry) = global
        .pointer_mut("/mcpServers")
        .and_then(|s| s.as_object_mut())
    else {
        return;
    };
    for (name, server) in servers {
        let mut entry = serde_json::Map::new();
        for key in ["command", "args", "env"] {
            if let Some(v) = server.get(key) {
                entry.insert(key.to_string(), v.clone());
            }
        }
        entry.insert("disabled".to_string(), serde_json::Value::Bool(false));
        registry.insert(name.clone(), serde_json::Value::Object(entry));
    }
    if let Some(parent) = global_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(&global) {
        Ok(text) => {
            if let Err(error) = std::fs::write(&global_path, text) {
                warn!(%error, "failed to write agy mcp_config.json");
            }
        }
        Err(error) => warn!(%error, "failed to serialize agy mcp_config.json"),
    }
}
