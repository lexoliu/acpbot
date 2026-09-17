//! A minimal Chrome DevTools Protocol client — just enough to drive a
//! real Chrome for the agent's browser tools, and nothing more.
//!
//! Stealth is the reason this exists instead of a crate: everything the
//! client sends is written here, so the session only ever enables the
//! domains we choose. `Runtime.enable` — the channel anti-bot walls
//! (rebrowser-style) probe to unmask CDP drivers — is never sent, and
//! no third-party driver library injects it on frame attach behind our
//! back. Input goes through `Input.dispatch*`, real trusted events.
//!
//! The wire is one browser-level WebSocket: commands carry an `id`, and
//! page-scoped commands add the flat-mode `sessionId` returned by
//! `Target.attachToTarget`. A reader task demultiplexes responses to
//! their callers and files the few events we subscribe to (`Log`,
//! `Network`) into bounded buffers the tools drain.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_lock::Mutex;
use async_tungstenite::tungstenite::Message;
use futures_lite::{StreamExt, future};
use serde_json::{Value, json};
use tracing::{debug, trace};

use crate::error::BrowserError;

/// How long one CDP call may sit unanswered before the connection is
/// declared dead. Generous — a wedged renderer shouldn't kill us fast,
/// but a dead one must not hang a tool call forever.
const CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Event buffers the reader task fills; tools drain them.
#[derive(Default)]
pub struct Buffers {
    /// `Log.entryAdded`: (level, text, source url).
    pub console: VecDeque<(String, String, String)>,
    /// `Network.requestWillBeSent` awaiting a response, by request id.
    requests: HashMap<String, (String, String)>,
    /// Completed exchanges, oldest first: `status METHOD url`.
    pub network: VecDeque<String>,
}

const BUFFER_CAP: usize = 200;

/// The id → waiter map shared between `call` and the reader task.
type Pending = Arc<Mutex<HashMap<u64, async_channel::Sender<Result<Value, BrowserError>>>>>;

/// A live connection to Chrome's browser-level DevTools socket.
pub struct Cdp {
    /// Next command id.
    next: AtomicU64,
    /// The ws write half — serialized so frames never interleave.
    write: Mutex<async_tungstenite::WebSocketSender<async_net::TcpStream>>,
    /// In-flight callers waiting on their id.
    pending: Pending,
    /// Filled by the reader task; cloned into the browser state.
    pub buffers: Arc<Mutex<Buffers>>,
    /// False once the reader task exits — the socket is gone.
    alive: Arc<AtomicBool>,
}

impl Cdp {
    /// Open the DevTools websocket and start the demultiplexing reader.
    ///
    /// # Errors
    /// [`BrowserError::Io`]/[`BrowserError::Handshake`] on connect or
    /// handshake failure.
    pub async fn connect(ws_url: &str) -> Result<Self, BrowserError> {
        // ws://127.0.0.1:PORT/devtools/browser/UUID — host:port + path.
        let without_scheme = ws_url
            .strip_prefix("ws://")
            .ok_or_else(|| BrowserError::Protocol(format!("bad ws url {ws_url}")))?;
        let (host_port, _path) = without_scheme
            .split_once('/')
            .ok_or_else(|| BrowserError::Protocol(format!("bad ws url {ws_url}")))?;
        let tcp = async_net::TcpStream::connect(host_port).await?;
        // No Origin header is sent, which Chrome's origin check accepts;
        // `--remote-allow-origins=*` covers the case anyway.
        let (ws, _response) = async_tungstenite::client_async(ws_url, tcp)
            .await
            .map_err(|error| BrowserError::Handshake(error.to_string()))?;
        let (write, mut read) = ws.split();

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let buffers: Arc<Mutex<Buffers>> = Arc::new(Mutex::new(Buffers::default()));
        let alive = Arc::new(AtomicBool::new(true));

        let reader = {
            let pending = Arc::clone(&pending);
            let buffers = Arc::clone(&buffers);
            let alive = Arc::clone(&alive);
            async move {
                while let Some(frame) = read.next().await {
                    let Ok(Message::Text(text)) = frame.map_err(|e| debug!(%e, "cdp read")) else {
                        continue;
                    };
                    let Ok(message) = serde_json::from_str::<Value>(&text) else {
                        continue;
                    };
                    if let Some(id) = message.get("id").and_then(Value::as_u64) {
                        let sender = pending.lock().await.remove(&id);
                        let response = if let Some(error) = message.get("error") {
                            Err(BrowserError::Protocol(format!(
                                "{}: {}",
                                message
                                    .get("method")
                                    .and_then(Value::as_str)
                                    .unwrap_or("command"),
                                error
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("cdp error")
                            )))
                        } else {
                            Ok(message.get("result").cloned().unwrap_or(Value::Null))
                        };
                        if let Some(sender) = sender {
                            let _ = sender.try_send(response);
                        }
                        continue;
                    }
                    if let Some(method) = message.get("method").and_then(Value::as_str) {
                        handle_event(method, &message, &buffers).await;
                    }
                }
                // Socket closed: fail every waiter and mark the
                // connection dead so the next tool call relaunches.
                alive.store(false, Ordering::SeqCst);
                pending.lock().await.clear();
            }
        };
        executor_core::spawn(reader).detach();

        Ok(Self {
            next: AtomicU64::new(1),
            write: Mutex::new(write),
            pending,
            buffers,
            alive,
        })
    }

    /// Whether the socket is still connected.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// One command. `session` scopes it to an attached page target;
    /// `None` talks to the browser itself (`Target.*`, `Browser.*`).
    ///
    /// # Errors
    /// [`BrowserError::Protocol`] on a CDP error response,
    /// [`BrowserError::Send`] on send failure, [`BrowserError::WentAway`]
    /// when the socket dies mid-call, [`BrowserError::Timeout`] after
    /// [`CALL_TIMEOUT`].
    pub async fn call(
        &self,
        session: Option<&str>,
        method: &str,
        params: Value,
    ) -> Result<Value, BrowserError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let mut command = json!({"id": id, "method": method, "params": params});
        if let Some(session) = session {
            command["sessionId"] = json!(session);
        }
        let (tx, rx) = async_channel::bounded(1);
        self.pending.lock().await.insert(id, tx);
        {
            let mut write = self.write.lock().await;
            write
                .send(Message::Text(command.to_string().into()))
                .await
                .map_err(|error| BrowserError::Send(Box::new(error)))?;
        }
        let timed = async {
            async_io::Timer::after(CALL_TIMEOUT).await;
            Err(BrowserError::Timeout(method.to_string()))
        };
        let answered = async {
            match rx.recv().await {
                Ok(result) => result,
                Err(_) => Err(BrowserError::WentAway),
            }
        };
        let result = future::or(answered, timed).await;
        if matches!(result, Err(BrowserError::Timeout(_))) {
            self.pending.lock().await.remove(&id);
        }
        trace!(method, ok = result.is_ok(), "cdp call");
        result
    }
}

/// File the events we subscribe to into [`Buffers`]. Everything else —
/// frame attach chatter, screencast frames — is noise to us.
async fn handle_event(method: &str, message: &Value, buffers: &Arc<Mutex<Buffers>>) {
    let params = &message["params"];
    match method {
        "Log.entryAdded" => {
            let entry = &params["entry"];
            let mut locked = buffers.lock().await;
            locked.console.push_back((
                entry["level"].as_str().unwrap_or("info").to_string(),
                entry["text"].as_str().unwrap_or_default().to_string(),
                entry["url"].as_str().unwrap_or_default().to_string(),
            ));
            if locked.console.len() > BUFFER_CAP {
                locked.console.pop_front();
            }
        }
        "Network.requestWillBeSent" => {
            let request = &params["request"];
            let id = params["requestId"].as_str().unwrap_or_default().to_string();
            let line = (
                request["method"].as_str().unwrap_or("GET").to_string(),
                request["url"].as_str().unwrap_or_default().to_string(),
            );
            let mut locked = buffers.lock().await;
            if locked.requests.len() < BUFFER_CAP * 4 {
                locked.requests.insert(id, line);
            }
        }
        "Network.responseReceived" => {
            let id = params["requestId"].as_str().unwrap_or_default();
            let status = params["response"]["status"].as_u64().unwrap_or(0);
            let mut locked = buffers.lock().await;
            if let Some((verb, url)) = locked.requests.remove(id) {
                locked.network.push_back(format!("{status} {verb} {url}"));
                if locked.network.len() > BUFFER_CAP {
                    locked.network.pop_front();
                }
            }
        }
        _ => {}
    }
}

/// Read Chrome's `DevToolsActivePort` file in the profile dir — it
/// appears once the browser is ready and carries `<port>\n<ws path>`.
/// Polls briefly since Chrome writes it a moment after spawn.
///
/// # Errors
/// [`BrowserError::Timeout`] when Chrome never writes the file.
pub async fn devtools_ws_url(profile: &std::path::Path) -> Result<String, BrowserError> {
    let file = profile.join("DevToolsActivePort");
    for _ in 0..150 {
        if let Ok(text) = std::fs::read_to_string(&file)
            && let Some((port, path)) = text.split_once('\n')
        {
            return Ok(format!("ws://127.0.0.1:{}{}", port.trim(), path.trim()));
        }
        async_io::Timer::after(std::time::Duration::from_millis(100)).await;
    }
    Err(BrowserError::Timeout("DevToolsActivePort".to_string()))
}
