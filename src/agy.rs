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

use aither_acp::{
    AgentCapabilities, ContentBlock, ContentChunk, Implementation, InitializeResult,
    McpCapabilities, PROTOCOL_VERSION, PromptCapabilities, PromptParams, PromptResult,
    SessionCancelParams, SessionCapabilities, SessionCloseCapabilities, SessionCloseParams,
    SessionCloseResult, SessionConfigValue, SessionDeleteParams, SessionDeleteResult, SessionInfo,
    SessionListCapabilities, SessionListResult, SessionLoadParams, SessionLoadResult,
    SessionNewParams, SessionNewResult, SessionNotification, SessionResumeCapabilities,
    SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult, SessionUpdate,
    StopReason, TextContent, ToolCall, ToolCallStatus, ToolCallUpdate, ToolKind,
};
use aither_mcp::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
use aither_mcp::transport::{BidirectionalTransport, StdioTransport, Transport};
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

/// The bridge's whole state: sessions acpbot opened plus the child handle
/// a `session/cancel` kills mid-turn.
struct Bridge {
    sessions: HashMap<String, AgySession>,
    /// The running turn's agy child per session, so `session/cancel` —
    /// a notification, not a request — can kill it while the prompt
    /// request is still being serviced.
    running: HashMap<String, Arc<Mutex<Option<Child>>>>,
}

/// One ACP session backed by an agy conversation.
struct AgySession {
    /// The session's working directory — agy inherits it as its cwd.
    cwd: PathBuf,
    /// `--model` override from `session/set_config_option("model")`.
    model: Option<String>,
    /// `--mode` override from `session/set_mode` (accept-edits/plan only).
    mode: Option<String>,
    /// The agy conversation backing this session — known once the first
    /// turn's `init` event reports it.
    conversation: Option<String>,
}

/// Run the bridge: read ACP JSON-RPC from stdin until EOF.
pub fn run() -> Result<(), BridgeError> {
    futures_lite::future::block_on(serve()).map_err(BridgeError::from)
}

async fn serve() -> Result<(), aither_mcp::protocol::McpError> {
    let mut transport = StdioTransport::new();
    let mut bridge = Bridge {
        sessions: HashMap::new(),
        running: HashMap::new(),
    };
    loop {
        match transport.recv().await? {
            Some(JsonRpcMessage::Request(req)) => {
                let response = bridge.handle_request(req, &mut transport).await;
                transport.respond(response).await?;
            }
            Some(JsonRpcMessage::Notification(notif)) => bridge.handle_notification(&notif),
            Some(JsonRpcMessage::Response(_)) => {}
            None => break,
        }
    }
    // stdin closed: kill any in-flight agy child so none outlive the daemon.
    for child in bridge.running.values() {
        if let Some(mut c) = child.lock().expect("running").take() {
            let _ = c.kill();
        }
    }
    Ok(())
}

impl Bridge {
    /// Notifications: only `session/cancel` does anything — kill the turn's
    /// agy child; the turn loop notices the closed stream and answers the
    /// outstanding prompt request with `stop_reason: cancelled`.
    fn handle_notification(&self, notif: &JsonRpcNotification) {
        if notif.method == "session/cancel"
            && let Some(Ok(SessionCancelParams { session_id, .. })) =
                notif.params.clone().map(serde_json::from_value)
            && let Some(child) = self.running.get(&session_id)
            && let Some(mut c) = child.lock().expect("running").take()
        {
            let _ = c.kill();
        }
    }

    async fn handle_request(
        &mut self,
        req: JsonRpcRequest,
        transport: &mut StdioTransport,
    ) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => JsonRpcResponse::success(req.id, initialize_result()),
            "session/new" => self.session_new(req),
            "session/load" => self.session_restore(req, false),
            "session/resume" => self.session_restore(req, true),
            "session/set_mode" => self.session_set_mode(req),
            "session/set_config_option" => self.session_set_config(req),
            "session/prompt" => self.session_prompt(req, transport).await,
            "session/list" => self.session_list(req),
            "session/close" | "session/delete" => self.session_end(req),
            method => JsonRpcResponse::error(req.id, JsonRpcError::method_not_found(method)),
        }
    }

    /// `session/new`: register the chat MCP endpoint from the session's
    /// `.devin/mcp_config.json` with agy's global registry, then record a
    /// session keyed by a fresh id. agy itself spawns lazily at the first
    /// prompt so `set_mode`/`set_config_option` land before it does.
    fn session_new(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let params = match parse_params::<SessionNewParams>(&req) {
            Ok(p) => p,
            Err(resp) => return *resp,
        };
        mirror_mcp_servers(&params.cwd);
        let session_id = new_id();
        self.sessions.insert(
            session_id.clone(),
            AgySession {
                cwd: params.cwd,
                model: None,
                mode: None,
                conversation: None,
            },
        );
        JsonRpcResponse::success(
            req.id,
            SessionNewResult {
                session_id,
                ..SessionNewResult::default()
            },
        )
    }

    /// `session/load` and `session/resume`: rebind an acp session id to its
    /// recorded agy conversation so a recovery handoff can still ask the
    /// old session for its summary.
    fn session_restore(&mut self, req: JsonRpcRequest, resume: bool) -> JsonRpcResponse {
        let (session_id, cwd) = if resume {
            match parse_params::<SessionResumeParams>(&req) {
                Ok(p) => (p.session_id, p.cwd),
                Err(resp) => return *resp,
            }
        } else {
            match parse_params::<SessionLoadParams>(&req) {
                Ok(p) => (p.session_id, p.cwd),
                Err(resp) => return *resp,
            }
        };
        let map = conversation_map(&cwd);
        let Some(conversation) = map.get(&session_id).cloned() else {
            return JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_params(format!("unknown session: {session_id}")),
            );
        };
        mirror_mcp_servers(&cwd);
        self.sessions.insert(
            session_id,
            AgySession {
                cwd,
                model: None,
                mode: None,
                conversation: Some(conversation),
            },
        );
        if resume {
            JsonRpcResponse::success(req.id, SessionResumeResult::default())
        } else {
            JsonRpcResponse::success(req.id, SessionLoadResult::default())
        }
    }

    /// `session/set_mode`: acpbot's modes are devin-flavored (`bypass`);
    /// only agy's own `accept-edits`/`plan` become a `--mode` flag —
    /// everything else is accepted and ignored.
    fn session_set_mode(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        match parse_params::<SessionSetModeParams>(&req) {
            Ok(p) => {
                if let Some(session) = self.sessions.get_mut(&p.session_id) {
                    session.mode = Some(p.mode_id);
                }
                JsonRpcResponse::success(req.id, SessionSetModeResult::default())
            }
            Err(resp) => *resp,
        }
    }

    /// `session/set_config_option`: `model` becomes `--model`; other keys
    /// are accepted and ignored.
    fn session_set_config(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        match parse_params::<SessionSetConfigOptionParams>(&req) {
            Ok(p) => {
                if let Some(session) = self.sessions.get_mut(&p.session_id)
                    && p.config_id == "model"
                    && let SessionConfigValue::Select { value, .. } = p.value
                {
                    session.model = Some(value);
                }
                JsonRpcResponse::success(req.id, SessionSetConfigOptionResult::default())
            }
            Err(resp) => *resp,
        }
    }

    fn session_list(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let sessions = self
            .sessions
            .iter()
            .map(|(id, s)| SessionInfo {
                session_id: id.clone(),
                cwd: s.cwd.clone(),
                additional_directories: Vec::new(),
                title: None,
                updated_at: None,
                meta: None,
                extra: BTreeMap::new(),
            })
            .collect();
        JsonRpcResponse::success(
            req.id,
            SessionListResult {
                sessions,
                ..SessionListResult::default()
            },
        )
    }

    /// `session/close` + `session/delete`: end the session — kill its agy
    /// child if a turn is running, drop the state. The conversation map is
    /// kept: a later `session/load` may still need the recovery handle.
    fn session_end(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        let session_id = if req.method == "session/delete" {
            match parse_params::<SessionDeleteParams>(&req) {
                Ok(p) => p.session_id,
                Err(resp) => return *resp,
            }
        } else {
            match parse_params::<SessionCloseParams>(&req) {
                Ok(p) => p.session_id,
                Err(resp) => return *resp,
            }
        };
        if let Some(child) = self.running.remove(&session_id)
            && let Some(mut c) = child.lock().expect("running").take()
        {
            let _ = c.kill();
        }
        if self.sessions.remove(&session_id).is_none() {
            return JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_params(format!("session not found: {session_id}")),
            );
        }
        if req.method == "session/delete" {
            JsonRpcResponse::success(req.id, SessionDeleteResult::default())
        } else {
            JsonRpcResponse::success(req.id, SessionCloseResult::default())
        }
    }

    /// `session/prompt`: spawn `agy` for this one turn, feed it the prompt
    /// as a single `user` stream-json event, translate its `step_update`
    /// stream into `session/update` notifications, and answer with the
    /// turn's stop reason when `result` arrives.
    ///
    /// While the turn runs, stdin keeps being read so `session/cancel`
    /// lands mid-turn; other requests queue and are answered after.
    async fn session_prompt(
        &mut self,
        req: JsonRpcRequest,
        transport: &mut StdioTransport,
    ) -> JsonRpcResponse {
        let params = match parse_params::<PromptParams>(&req) {
            Ok(p) => p,
            Err(resp) => return *resp,
        };
        // The session leaves the map for the turn's duration so `self`
        // stays free for transport I/O; it goes back once the turn ends.
        let Some(mut session) = self.sessions.remove(&params.session_id) else {
            return JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_params(format!("session not found: {}", params.session_id)),
            );
        };

        let stop = self.run_turn(transport, &mut session, &params).await;
        self.sessions.insert(params.session_id.clone(), session);

        JsonRpcResponse::success(
            req.id,
            PromptResult {
                stop_reason: stop,
                meta: None,
            },
        )
    }

    /// One turn of `agy` for `session`, streaming updates to `transport`.
    /// Returns the turn's stop reason.
    async fn run_turn(
        &mut self,
        transport: &mut StdioTransport,
        session: &mut AgySession,
        params: &PromptParams,
    ) -> StopReason {
        let mut args = vec![
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ];
        if let Some(model) = &session.model {
            args.extend(["--model".to_string(), model.clone()]);
        }
        if let Some(mode) = &session.mode
            && matches!(mode.as_str(), "accept-edits" | "plan")
        {
            args.extend(["--mode".to_string(), mode.clone()]);
        }
        if let Some(conversation) = &session.conversation {
            args.extend(["--conversation".to_string(), conversation.clone()]);
        }
        args.extend(["--print".to_string(), String::new()]);

        let mut child = match Command::new(agy_bin())
            .args(&args)
            .current_dir(&session.cwd)
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
        let text = prompt_text(&params.prompt);
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
        // channel so the turn loop can race it against client input. `None`
        // on the channel marks end-of-stream.
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

        let cell = Arc::new(Mutex::new(Some(child)));
        self.running.insert(params.session_id.clone(), cell.clone());
        let mut cancelled = false;
        let mut deferred = Vec::new();

        /// Which side of the turn's `or` produced a value.
        enum Step {
            /// One stdout line, or `None` at end-of-stream.
            Line(Option<String>),
            /// A client message arrived mid-turn.
            Client(std::result::Result<Option<JsonRpcMessage>, aither_mcp::protocol::McpError>),
        }

        let stop = loop {
            let step = or(
                async { Step::Line(lines_rx.recv().await.unwrap_or(None)) },
                async { Step::Client(transport.recv().await) },
            )
            .await;
            match step {
                Step::Line(Some(line)) => {
                    if let Some(stop) =
                        agy_line(transport, session, &params.session_id, &line).await
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
                Step::Client(Ok(Some(JsonRpcMessage::Notification(notif)))) => {
                    if notif.method == "session/cancel" {
                        cancelled = true;
                        if let Some(mut c) = cell.lock().expect("running").take() {
                            let _ = c.kill();
                        }
                    }
                }
                Step::Client(Ok(Some(msg))) => deferred.push(msg),
                // The daemon went away mid-turn — kill the child; nobody
                // will read the response.
                Step::Client(Ok(None) | Err(_)) => {
                    if let Some(mut c) = cell.lock().expect("running").take() {
                        let _ = c.kill();
                    }
                    break StopReason::Cancelled;
                }
            }
        };

        // Reap the child — normally it has already exited after `result`;
        // bound the wait so a lingering process cannot hang the loop.
        if let Some(mut c) = cell.lock().expect("running").take() {
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
        self.running.remove(&params.session_id);

        for msg in deferred {
            match msg {
                JsonRpcMessage::Request(req) => {
                    let response = Box::pin(self.handle_request(req, transport)).await;
                    if transport.respond(response).await.is_err() {
                        break;
                    }
                }
                JsonRpcMessage::Notification(notif) => self.handle_notification(&notif),
                JsonRpcMessage::Response(_) => {}
            }
        }

        stop
    }
}

/// Translate one agy NDJSON line into `session/update` notifications.
/// Returns the turn's stop reason once the `result` event arrives.
async fn agy_line(
    transport: &mut StdioTransport,
    session: &mut AgySession,
    session_id: &str,
    line: &str,
) -> Option<StopReason> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        debug!("unparsed agy line: {}", &line[..line.len().min(200)]);
        return None;
    };
    match value.get("event").and_then(|e| e.as_str()) {
        Some("init") => {
            // First sight of the conversation id: bind it so later turns —
            // and a later `session/load` after a crash — resume the same
            // conversation.
            if let Some(conversation_id) = value.get("conversation_id").and_then(|c| c.as_str())
                && session.conversation.as_deref() != Some(conversation_id)
            {
                session.conversation = Some(conversation_id.to_string());
                let mut map = conversation_map(&session.cwd);
                map.insert(session_id.to_string(), conversation_id.to_string());
                save_conversation_map(&session.cwd, &map);
            }
            None
        }
        Some("step_update") => {
            let update = step_update(value.get("step_update")?)?;
            let notif = JsonRpcNotification::with_params(
                "session/update",
                SessionNotification {
                    session_id: session_id.to_string(),
                    update,
                    meta: None,
                    extra: BTreeMap::new(),
                },
            );
            let _ = transport.notify(notif).await;
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

/// One `step_update` payload → at most one ACP update.
fn step_update(step: &serde_json::Value) -> Option<SessionUpdate> {
    let step_type = step.get("step_type")?.as_str()?;
    match step_type {
        "agent_response" => text_chunk(step, false),
        "agent_thought" | "thinking" => text_chunk(step, true),
        "tool" => {
            let id = step.get("step_index")?.as_u64()?.to_string();
            let name = step
                .get("tool_name")
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            // Classify agy's command tools so the client's blocked-tool
            // rule works on `kind` — harness-agnostic — rather than these
            // agy-specific names.
            let kind = crate::handler::MAIN_BLOCKED_TOOLS
                .contains(&name)
                .then_some(ToolKind::Execute);
            if step.get("state").and_then(|s| s.as_str()) == Some("ACTIVE") {
                Some(SessionUpdate::ToolCall(ToolCall {
                    tool_call_id: id,
                    title: name.to_string(),
                    kind,
                    status: Some(ToolCallStatus::InProgress),
                    content: Vec::new(),
                    locations: Vec::new(),
                    raw_input: step
                        .get("tool_info")
                        .and_then(|i| i.get("parameters"))
                        .cloned(),
                    raw_output: None,
                    meta: None,
                }))
            } else {
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id,
                    status: Some(ToolCallStatus::Completed),
                    title: None,
                    kind,
                    content: None,
                    locations: None,
                    raw_input: None,
                    raw_output: step.get("tool_info").and_then(|i| i.get("output")).cloned(),
                    meta: None,
                }))
            }
        }
        _ => None,
    }
}

/// A text delta as an `agent_message_chunk`/`agent_thought_chunk` update.
fn text_chunk(step: &serde_json::Value, thought: bool) -> Option<SessionUpdate> {
    let text = step.get("text_delta")?.as_str()?.to_string();
    let chunk = ContentChunk {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
            meta: None,
        }),
        message_id: None,
        meta: None,
    };
    Some(if thought {
        SessionUpdate::AgentThoughtChunk(chunk)
    } else {
        SessionUpdate::AgentMessageChunk(chunk)
    })
}

/// The ACP `initialize` result: agy loads sessions, takes text prompts,
/// and honors `session/resume`/`close`/`list` — no image/audio prompts.
fn initialize_result() -> InitializeResult {
    InitializeResult {
        protocol_version: PROTOCOL_VERSION,
        agent_capabilities: AgentCapabilities {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                image: false,
                audio: false,
                embedded_context: false,
            },
            mcp_capabilities: McpCapabilities::default(),
            session_capabilities: SessionCapabilities {
                resume: Some(SessionResumeCapabilities::default()),
                close: Some(SessionCloseCapabilities::default()),
                list: Some(SessionListCapabilities::default()),
                ..SessionCapabilities::default()
            },
            ..AgentCapabilities::default()
        },
        agent_info: Some(Implementation {
            name: "acpbot agy-bridge".to_string(),
            title: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        auth_methods: Vec::new(),
        meta: None,
    }
}

/// The concatenated text of a prompt's text blocks; image/audio payloads
/// are not sent to agy (its stream-json input is text) — the event JSON
/// already carries their `inbox/` paths for the file tools.
fn prompt_text(prompt: &[ContentBlock]) -> String {
    prompt
        .iter()
        .filter_map(|block| {
            if let ContentBlock::Text(text) = block {
                Some(text.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse `req.params` into `P`, producing the error response on failure.
fn parse_params<P: serde::de::DeserializeOwned>(
    req: &JsonRpcRequest,
) -> std::result::Result<P, Box<JsonRpcResponse>> {
    match req.params.clone().map(serde_json::from_value).transpose() {
        Ok(Some(p)) => Ok(p),
        _ => Err(Box::new(JsonRpcResponse::error(
            req.id.clone(),
            JsonRpcError::invalid_params(format!("bad params for {}", req.method)),
        ))),
    }
}

/// A fresh session id — opaque; the agy conversation id it binds is what
/// matters.
fn new_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!("{:08x}-{:032x}", std::process::id(), nanos)
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
