//! Per-chat agent lifecycle.
//!
//! Every chat gets its own `devin acp` child process and ACP session, driven
//! by an actor that owns the chat's event queue. Per-process-per-chat keeps
//! failure isolated and lets the session's working directory carry the
//! chat-specific `.devin/mcp_config.json` — the channel through which the
//! `chat` MCP tools reach the agent.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::{AgentError, SenderError};
use aither_acp::{
    AcpClient, AgentCapabilities, AudioContent, ClientError, ContentBlock, ImageContent,
    PromptCapabilities, PromptResult, TextContent,
};
use async_channel::{Receiver, Sender as ChanSender};
use base64::Engine as _;
use tracing::{debug, error, info, warn};

use crate::chat::{ChatEvent, ChatKey};
use crate::config::{AgentConfig, AgentIsolation};
use crate::handler::BotClientHandler;
use crate::mcpserver;
use crate::sandbox::{AgentRuntime, BridgeTarget};
use crate::sender::Sender;

/// Builds a platform [`Sender`] bound to one chat. The channel carries the
/// chat's "the agent spoke" signal shared by all senders of that chat; the
/// `AtomicI64` is the forum-topic cell and the `AtomicU64` the last-action
/// timestamp (epoch ms) every sender of the chat shares.
pub type SenderFactory = Box<
    dyn Fn(&ChatKey, ChanSender<()>, Arc<AtomicI64>, Arc<AtomicU64>) -> Result<Sender, SenderError>
        + Send
        + Sync,
>;

/// What every chat actor shares: agent launch config, paths, the sticker
/// pack directory, and a factory that binds a platform [`Sender`] to a chat.
pub struct AgentShared {
    /// How to spawn and configure the agent process.
    pub agent: AgentConfig,
    /// The `acpbot` binary the agent's `.devin/mcp_config.json` invokes for
    /// `mcp-bridge`. Normally the current executable.
    pub bridge_bin: PathBuf,
    /// Daemon data root.
    pub data_dir: PathBuf,
    /// The sticker pack directory — re-scanned per tool call, and writable
    /// by the agent so it can extend its own pack.
    pub sticker_dir: PathBuf,
    /// Persona text injected into every chat's `AGENTS.md`.
    pub persona: Option<String>,
    /// Imported foreign sticker sets — bot-global, persisted under
    /// `data_dir`, shared by every chat's `send_sticker`.
    pub sticker_library: Arc<crate::stickerlib::StickerLibrary>,
    /// Builds a platform sender bound to a chat.
    pub sender_for: SenderFactory,
    /// Where actors report `(chat, session_id)` once a session exists; the
    /// dispatcher persists it for `session/load` on restart.
    pub session_updates: ChanSender<(ChatKey, String)>,
}

/// Media above this many bytes stays an `inbox/` path in the event JSON
/// rather than also being inlined as an `image`/`audio` content block —
/// keeps a huge attachment from dominating the prompt.
const MEDIA_INLINE_LIMIT: u64 = 8 * 1024 * 1024;

/// Routes incoming events to per-chat actors, spawning one on first sight.
pub struct Dispatcher {
    shared: Arc<AgentShared>,
    actors: HashMap<ChatKey, (ChanSender<ChatEvent>, executor_core::AnyExecutorTask<()>)>,
    /// chat key slug → ACP session id, loaded from `sessions.json`.
    known_sessions: HashMap<String, String>,
    /// Receives session ids reported by actors.
    session_ids: Receiver<(ChatKey, String)>,
    /// Cloned into every actor; dropping the sender ends their in-flight
    /// turns so a daemon shutdown never leaves a sandboxed agent running.
    shutdown: (Option<ChanSender<()>>, Receiver<()>),
}

impl Dispatcher {
    /// Load the session registry and build the dispatcher.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent: AgentConfig,
        bridge_bin: PathBuf,
        data_dir: PathBuf,
        sticker_dir: PathBuf,
        persona: Option<String>,
        sender_for: SenderFactory,
        sticker_library: Arc<crate::stickerlib::StickerLibrary>,
    ) -> Self {
        let (session_updates, session_ids) = async_channel::unbounded();
        let known_sessions = load_sessions(&data_dir);
        let (shutdown_tx, shutdown_rx) = async_channel::unbounded();
        Self {
            shared: Arc::new(AgentShared {
                agent,
                bridge_bin,
                data_dir,
                sticker_dir,
                persona,
                sticker_library,
                sender_for,
                session_updates,
            }),
            actors: HashMap::new(),
            known_sessions,
            session_ids,
            shutdown: (Some(shutdown_tx), shutdown_rx),
        }
    }

    /// Consume events forever, routing each to its chat's actor.
    pub async fn run(mut self, events: Receiver<ChatEvent>) {
        loop {
            let event = futures_lite::future::or(
                async { events.recv().await.map(|e| Routed::Event(Box::new(e))) },
                async { self.session_ids.recv().await.map(Routed::SessionId) },
            )
            .await;
            match event {
                Ok(Routed::Event(event)) => self.dispatch(event).await,
                Ok(Routed::SessionId((key, sid))) => {
                    self.known_sessions.insert(key.slug(), sid);
                    save_sessions(&self.shared.data_dir, &self.known_sessions);
                }
                Err(_) => break,
            }
        }

        // The event channel closed: the daemon is shutting down. Closing the
        // shutdown channel interrupts in-flight turns; dropping each actor's
        // sender ends its run loop; awaiting its task lets the actor drop its
        // runtime — a heel Sandbox kills the agent process on the way out.
        drop(self.shutdown.0.take());
        for (tx, task) in self.actors.drain().map(|(_, entry)| entry) {
            drop(tx);
            task.await;
        }
    }

    async fn dispatch(&mut self, event: Box<ChatEvent>) {
        let event = *event;
        let key = ChatKey {
            platform: event.platform,
            id: event.chat.clone(),
        };
        if let Some((tx, _)) = self.actors.get(&key)
            && tx.send(event.clone()).await.is_ok()
        {
            return;
        }

        // No actor yet (or it died): spawn one.
        let (tx, rx) = async_channel::unbounded();
        // Capacity 1: a full channel is the "the agent already spoke
        // this turn" flag, and the drain at turn start resets it.
        let (spoke_tx, spoke_rx) = async_channel::bounded(1);
        let last_action = Arc::new(AtomicU64::new(crate::sender::epoch_ms()));
        let (restart_tx, restart_rx) = async_channel::unbounded();
        let actor = ChatActor {
            key: key.clone(),
            shared: self.shared.clone(),
            history: Arc::new(crate::history::History::open(crate::history::history_path(
                &self.shared.data_dir.join("chats").join(key.slug()),
            ))),
            client: None,
            session_id: self.known_sessions.get(&key.slug()).cloned(),
            bridge_target: None,
            runtime: None,
            child: None,
            thread: Arc::new(AtomicI64::new(0)),
            prompt_caps: PromptCapabilities::default(),
            compact_supported: Arc::new(AtomicBool::new(false)),
            spoke_tx,
            spoke_rx,
            last_action,
            restart_tx,
            restart_rx,
            shutdown_rx: self.shutdown.1.clone(),
        };
        let task = executor_core::spawn(actor.run(rx));
        self.actors.insert(key.clone(), (tx.clone(), task));
        if tx.send(event).await.is_err() {
            error!(%key, "fresh chat actor refused first event");
        }
    }
}

enum Routed {
    Event(Box<ChatEvent>),
    SessionId((ChatKey, String)),
}

/// What [`wait_event`] resolved to.
enum Waited {
    /// An event arrived.
    Event(Box<ChatEvent>),
    /// The idle window expired with no event.
    Idle,
    /// The event channel closed — the daemon is shutting down.
    Closed,
}

/// Wait for the next event or `idle` of silence, whichever comes first.
async fn wait_event(rx: &Receiver<ChatEvent>, idle: Duration) -> Waited {
    futures_lite::future::or(
        async {
            match rx.recv().await {
                Ok(event) => Waited::Event(Box::new(event)),
                Err(_) => Waited::Closed,
            }
        },
        async {
            async_io::Timer::after(idle).await;
            Waited::Idle
        },
    )
    .await
}

/// Ends a turn's typing task when dropped.
struct TurnTyping(async_channel::Sender<()>);

impl Drop for TurnTyping {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// One chat's agent process, session, and event queue.
struct ChatActor {
    key: ChatKey,
    shared: Arc<AgentShared>,
    /// The chat's transcript log — inbound events land here from `run`,
    /// outbound actions from the `Sender`, and the `history`/
    /// `search_history` tools read it back.
    history: Arc<crate::history::History>,
    client: Option<AcpClient<BotClientHandler>>,
    session_id: Option<String>,
    /// The bound MCP endpoint + the argument `mcp-bridge` is configured with.
    bridge_target: Option<String>,
    /// The chat's isolation runtime (sandbox/container), once created.
    runtime: Option<AgentRuntime>,
    /// The sandboxed child handle of the current agent process, if any.
    child: Option<heel::Child>,
    /// Forum topic the in-flight turn's events arrived in (`0` = none);
    /// every `Sender` built for this chat reads it so sends land in the
    /// topic the user addressed.
    thread: Arc<AtomicI64>,
    /// Shared with every `Sender` of this chat: receives a unit each time the
    /// agent produces outbound output.
    spoke_tx: ChanSender<()>,
    /// Drained per turn to stop the typing indicator on the first send.
    spoke_rx: Receiver<()>,
    /// Epoch ms of the agent's last outbound action — every sender writes
    /// it through `spoke`; the nudge watchdog reads it to measure silence.
    last_action: Arc<AtomicU64>,
    /// Closes when the daemon shuts down; an in-flight turn aborts on it so
    /// the runtime (and its sandboxed agent) is dropped, not orphaned.
    shutdown_rx: Receiver<()>,
    /// Handed to the `restart` chat tool — the agent posts to it to be
    /// reincarnated after the current turn.
    restart_tx: ChanSender<()>,
    /// Drained after each turn: a queued unit means the agent asked to be
    /// restarted so new skills/instructions load.
    restart_rx: Receiver<()>,
    /// Content types the agent declared at `initialize` — gates whether
    /// media reaches it as `image`/`audio` blocks or only an `inbox/` path.
    prompt_caps: PromptCapabilities,
    /// Set by the session's `available_commands_update` — `true` when the
    /// harness handles a `compact` command, so idle compaction is a local
    /// operation rather than text the model would answer in the chat.
    compact_supported: Arc<AtomicBool>,
}

impl ChatActor {
    /// The chat's working directory (`<data_dir>/chats/<slug>/`).
    fn cwd(&self) -> PathBuf {
        self.shared.data_dir.join("chats").join(self.key.slug())
    }

    /// The IPC socket the MCP bridge connects to.
    fn socket_path(&self) -> PathBuf {
        self.shared
            .data_dir
            .join("run")
            .join(self.key.slug() + ".sock")
    }

    /// The `command`/`args` `mcp_config.json` should spawn to reach this
    /// chat's MCP endpoint — the host `acpbot` binary for `none`/`native`,
    /// the configured in-container command for `docker`.
    fn mcp_server_entry(&self) -> Result<(String, Vec<String>), AgentError> {
        let target = self.bridge_target.clone().expect("bound first");
        match &self.shared.agent.isolation {
            AgentIsolation::Docker(docker) => {
                let (command, prefix) = docker.bridge_command.split_first().ok_or_else(|| {
                    AgentError::Other("[agent.isolation] bridge_command is empty".to_string())
                })?;
                let mut args: Vec<String> = prefix.to_vec();
                args.push(target);
                Ok((command.clone(), args))
            }
            _ => Ok((
                self.shared.bridge_bin.to_string_lossy().into_owned(),
                vec!["mcp-bridge".to_string(), target],
            )),
        }
    }

    /// The transcript file inside the chat cwd.
    fn transcript_path(&self) -> PathBuf {
        self.cwd().join("transcript.log")
    }

    async fn run(mut self, rx: Receiver<ChatEvent>) {
        // `true` once this quiet stretch already spent its one compaction —
        // stays set until the next event arrives so idle time never stacks
        // repeated compactions.
        let mut idle_compacted = false;
        let idle = self.shared.agent.idle_compact();
        // Events the in-flight turn pulled off `rx` but didn't consume —
        // they seed the next batch so ordering stays FIFO.
        let mut pending: Vec<ChatEvent> = Vec::new();
        loop {
            let mut batch = std::mem::take(&mut pending);
            if batch.is_empty() {
                let first = if idle_compacted || idle.is_zero() {
                    match rx.recv().await {
                        Ok(event) => event,
                        Err(_) => return,
                    }
                } else {
                    match wait_event(&rx, idle).await {
                        Waited::Event(event) => *event,
                        Waited::Closed => return,
                        Waited::Idle => {
                            if let Err(error) = self.compact().await {
                                warn!(chat = %self.key, %error, "idle compaction failed");
                            }
                            idle_compacted = true;
                            continue;
                        }
                    }
                };
                self.history.append_event(&first);
                batch.push(first);
            }
            idle_compacted = false;
            // Coalesce events that arrived while a turn was in flight.
            while let Ok(event) = rx.try_recv() {
                self.history.append_event(&event);
                batch.push(event);
            }
            if let Err(error) = self.turn(&batch, &rx, &mut pending).await {
                error!(chat = %self.key, %error, "turn failed");
            }
        }
    }

    /// Ask the agent to compact the session's history after a quiet
    /// stretch: prompt caches outlive the configured idle window, so the
    /// compaction itself is cheap, and the smaller context is what the
    /// user's next message pays for. No typing indicator, no fallback ack
    /// — maintenance the chat never sees.
    ///
    /// No-ops when the session never advertised a `compact` command —
    /// `/compact` is devin's spelling; a harness without it would treat
    /// the prompt as ordinary text and could answer in the chat.
    async fn compact(&mut self) -> Result<(), AgentError> {
        let (Some(client), Some(session_id)) = (self.client.clone(), self.session_id.clone())
        else {
            return Ok(());
        };
        if !self.compact_supported.load(Ordering::Relaxed) {
            debug!(chat = %self.key, "idle; agent has no compact command, skipping");
            return Ok(());
        }
        info!(chat = %self.key, "idle; compacting session");
        let result = client
            .prompt(
                &session_id,
                vec![ContentBlock::Text(TextContent {
                    text: "/compact".to_string(),
                    annotations: None,
                })],
            )
            .await?;
        debug!(chat = %self.key, stop = ?result.stop_reason, "idle compaction done");
        Ok(())
    }

    /// Deliver one batch of events to the agent as a single prompt.
    ///
    /// `rx` stays wired into the prompt wait so a `stop` event can cancel
    /// the turn mid-flight; non-stop events pulled that way land in
    /// `pending` and seed the caller's next batch.
    async fn turn(
        &mut self,
        batch: &[ChatEvent],
        rx: &Receiver<ChatEvent>,
        pending: &mut Vec<ChatEvent>,
    ) -> Result<(), AgentError> {
        let turn_started = Instant::now();
        self.thread.store(
            batch
                .iter()
                .filter_map(|event| event.thread_id)
                .next_back()
                .unwrap_or(0),
            Ordering::Relaxed,
        );
        self.ensure_ready().await?;

        let sender = (self.shared.sender_for)(
            &self.key,
            self.spoke_tx.clone(),
            self.thread.clone(),
            self.last_action.clone(),
        )?;

        // Pull every file the events carry into `inbox/` so the agent can
        // open the bytes; image payloads also go into the prompt as `image`
        // content blocks so vision-capable models see them directly.
        let chat_dir = self.cwd();
        let mut batch = batch.to_vec();
        for event in &mut batch {
            if let Some(media) = &mut event.media {
                media.file = sender.fetch_media(&media.file_id, &chat_dir).await?;
            }
            if let Some(sticker) = &mut event.sticker {
                sticker.file = sender.fetch_media(&sticker.file_id, &chat_dir).await?;
            }
        }

        let text = if batch.len() == 1 {
            batch[0].to_prompt_text()
        } else {
            serde_json::to_string(&batch).expect("events serialize")
        };
        let mut prompt = vec![ContentBlock::Text(TextContent {
            text,
            annotations: None,
        })];
        for file in batch.iter().flat_map(ChatEvent::files) {
            let is_image = self.prompt_caps.image && file.mime.starts_with("image/");
            let is_audio = self.prompt_caps.audio && file.mime.starts_with("audio/");
            if !is_image && !is_audio {
                continue;
            }
            let bytes = async_fs::read(chat_dir.join(&file.path)).await?;
            if bytes.len() as u64 > MEDIA_INLINE_LIMIT {
                continue;
            }
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            let mime_type = file.mime.clone();
            prompt.push(if is_image {
                ContentBlock::Image(ImageContent { data, mime_type })
            } else {
                ContentBlock::Audio(AudioContent { data, mime_type })
            });
        }

        let _typing = self.start_turn_typing(sender, turn_started);
        self.last_action
            .store(crate::sender::epoch_ms(), Ordering::Relaxed);

        let client = self.client.clone().expect("ensure_ready ran");
        let session_id = self.session_id.clone().expect("ensure_ready ran");

        let outcome = match self
            .prompt_with_nudges(&client, &session_id, prompt.clone(), rx, pending)
            .await
        {
            // A daemon shutdown must interrupt the prompt wait: dropping the
            // client, child, and runtime here is what lets a sandbox's Drop
            // kill the agent process instead of orphaning it.
            Ok(PromptEnd::Shutdown) => {
                self.client = None;
                if let Some(mut child) = self.child.take() {
                    let _ = child.kill();
                }
                self.runtime = None;
                return Ok(());
            }
            Ok(PromptEnd::Done(result)) => {
                debug_turn_end(&result);
                Ok(())
            }
            // The user said stop: the turn is already cancelled and its
            // remaining work discarded — a quiet end, not a failure.
            Ok(PromptEnd::Stopped) => Ok(()),
            // A dead child fails every pending request with `Closed`: respawn
            // once, reload the session, and retry the prompt once.
            Err(ClientError::Closed { .. }) => {
                warn!(chat = %self.key, "agent process died; respawning");
                self.client = None;
                if let Some(mut child) = self.child.take() {
                    let _ = child.kill();
                }
                self.ensure_ready().await?;
                let client = self.client.clone().expect("ensure_ready ran");
                let session_id = self.session_id.clone().expect("ensure_ready ran");
                match self
                    .prompt_with_nudges(&client, &session_id, prompt, rx, pending)
                    .await
                {
                    Ok(PromptEnd::Done(result)) => {
                        debug_turn_end(&result);
                        Ok(())
                    }
                    Ok(PromptEnd::Shutdown | PromptEnd::Stopped) => Ok(()),
                    Err(error) => Err(error.into()),
                }
            }
            Err(error) => Err(error.into()),
        };

        // The agent asked to be reincarnated this turn (new skills or edited
        // instructions): drop client, process, and runtime — the next event
        // respawns and `session/load` resumes the session. Drain every
        // queued request so duplicate calls don't cause repeated restarts.
        let mut restart = false;
        while self.restart_rx.try_recv().is_ok() {
            restart = true;
        }
        if restart {
            info!(chat = %self.key, "agent requested restart");
            self.client = None;
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
            }
            self.runtime = None;
        }
        outcome
    }

    /// Run one prompt to completion, enforcing the never-keep-them-waiting
    /// rule: whenever the agent stays silent past
    /// `[agent] nudge_after_secs`, the turn is cancelled and re-prompted
    /// with a `nudge` event so the agent sends the user an update and
    /// resumes its work. `0` disables the nudge entirely.
    ///
    /// While the prompt runs, `rx` stays watched: a `stop` event cancels
    /// the turn outright, and any other event lands in `pending` for the
    /// caller's next batch — pulled early instead of sitting in the
    /// channel, same FIFO order.
    async fn prompt_with_nudges(
        &mut self,
        client: &AcpClient<BotClientHandler>,
        session_id: &str,
        content: Vec<ContentBlock>,
        rx: &Receiver<ChatEvent>,
        pending: &mut Vec<ChatEvent>,
    ) -> Result<PromptEnd, ClientError> {
        let nudge_after = self.shared.agent.nudge_after();
        let mut prompt: Pin<
            Box<dyn Future<Output = Result<PromptResult, ClientError>> + Send + '_>,
        > = Box::pin(client.prompt(session_id, content));
        // Once `rx` closes no event can ever arrive — drop the arm rather
        // than spin on instant `Err`s.
        let mut events_open = true;
        loop {
            enum Race {
                Done(Result<PromptResult, ClientError>),
                Shutdown,
                Tick,
                Event(Box<ChatEvent>),
                EventsClosed,
            }
            let done = async { Race::Done(prompt.as_mut().await) };
            let wait = async {
                let _ = self.shutdown_rx.recv().await;
                Race::Shutdown
            };
            let tick = async {
                if nudge_after.is_zero() {
                    std::future::pending::<()>().await;
                }
                let silent_ms = crate::sender::epoch_ms()
                    .saturating_sub(self.last_action.load(Ordering::Relaxed));
                let remaining = nudge_after.saturating_sub(Duration::from_millis(silent_ms));
                async_io::Timer::after(remaining).await;
                Race::Tick
            };
            let incoming = async {
                if !events_open {
                    std::future::pending::<()>().await;
                }
                match rx.recv().await {
                    Ok(event) => Race::Event(Box::new(event)),
                    Err(_) => Race::EventsClosed,
                }
            };
            let race = futures_lite::future::or(
                futures_lite::future::or(done, wait),
                futures_lite::future::or(tick, incoming),
            )
            .await;
            match race {
                Race::Done(result) => return result.map(PromptEnd::Done),
                Race::Shutdown => return Ok(PromptEnd::Shutdown),
                Race::EventsClosed => events_open = false,
                Race::Event(event) => {
                    self.history.append_event(&event);
                    if event.is_stop() {
                        info!(chat = %self.key, "user said stop; cancelling turn");
                        cancel_and_settle(client, session_id, &mut prompt, &self.key).await;
                        return Ok(PromptEnd::Stopped);
                    }
                    pending.push(*event);
                }
                Race::Tick => {
                    let silent_ms = crate::sender::epoch_ms()
                        .saturating_sub(self.last_action.load(Ordering::Relaxed));
                    if silent_ms < nudge_after.as_millis() as u64 {
                        continue;
                    }
                    warn!(
                        chat = %self.key,
                        silent_ms, "agent silent past the nudge deadline; interrupting"
                    );
                    cancel_and_settle(client, session_id, &mut prompt, &self.key).await;
                    // The nudged turn gets a fresh silence budget — it must
                    // still speak within `nudge_after` or be interrupted
                    // again.
                    self.last_action
                        .store(crate::sender::epoch_ms(), Ordering::Relaxed);
                    prompt = Box::pin(client.prompt(
                        session_id,
                        vec![ContentBlock::Text(TextContent {
                            text: nudge_event_text(),
                            annotations: None,
                        })],
                    ));
                }
            }
        }
    }

    /// Bring up the chat MCP endpoint, the isolation runtime, the agent
    /// process, and the ACP session.
    async fn ensure_ready(&mut self) -> Result<(), AgentError> {
        if self.bridge_target.is_none() {
            self.bridge_target = Some(self.bind_mcp_endpoint().await?);
        }

        let cwd = self.cwd();
        std::fs::create_dir_all(&cwd)?;

        if self.runtime.is_none() {
            self.runtime = Some(
                AgentRuntime::create(
                    &self.shared.agent.isolation,
                    &self.shared.agent,
                    &cwd,
                    &self.socket_path(),
                    &self.shared.bridge_bin,
                    &self.shared.sticker_dir,
                )
                .await?,
            );
        }

        if self.client.is_some() {
            return Ok(());
        }

        let (mcp_command, mcp_args) = self.mcp_server_entry()?;
        prepare_chat_dir(&cwd, &mcp_command, &mcp_args, &self.shared)?;

        // The new session will re-advertise its commands; don't trust a
        // previous process's `compact` in the meantime.
        self.compact_supported.store(false, Ordering::Relaxed);
        let handler = BotClientHandler::new(self.transcript_path(), self.compact_supported.clone());
        let spawned = self
            .runtime
            .as_ref()
            .expect("created above")
            .spawn_agent(&self.shared.agent, &cwd, handler)
            .await?;
        let client = spawned.client;
        self.child = spawned.child;
        executor_core::spawn(spawned.connection).detach();

        let init = client.initialize().await?;
        self.prompt_caps = init.agent_capabilities.prompt_capabilities.clone();
        info!(
            chat = %self.key,
            agent = init.agent_info.as_ref().map_or("?", |i| i.name.as_str()),
            image = self.prompt_caps.image,
            audio = self.prompt_caps.audio,
            "agent initialized"
        );

        // Resume the persisted session when the agent supports it; otherwise
        // open a fresh one.
        let session_id = match self.session_id.take() {
            Some(sid) => match self
                .restore_session(&client, &init.agent_capabilities, &sid, &cwd)
                .await
            {
                Some(sid) => sid,
                None => self.new_session(&client, &cwd).await?,
            },
            None => self.new_session(&client, &cwd).await?,
        };

        if let Err(error) = client.set_mode(&session_id, &self.shared.agent.mode).await {
            warn!(chat = %self.key, %error, "set_mode failed");
        }
        if let Err(error) = client
            .set_config_option(&session_id, "model", self.shared.agent.model.clone())
            .await
        {
            warn!(chat = %self.key, %error, "set_config_option(model) failed");
        }

        self.session_id = Some(session_id.clone());
        self.client = Some(client);
        let _ = self
            .shared
            .session_updates
            .try_send((self.key.clone(), session_id));
        Ok(())
    }

    /// Bind this chat's MCP endpoint (unix socket or loopback TCP) and
    /// return the `mcp-bridge` argument the generated `mcp_config.json`
    /// carries.
    async fn bind_mcp_endpoint(&mut self) -> Result<String, AgentError> {
        let make_tools = {
            let sender = (self.shared.sender_for)(
                &self.key,
                self.spoke_tx.clone(),
                self.thread.clone(),
                self.last_action.clone(),
            )?
            .with_history(self.history.clone());
            let sticker_dir = self.shared.sticker_dir.clone();
            let sticker_library = self.shared.sticker_library.clone();
            let restart = self.restart_tx.clone();
            let history = self.history.clone();
            move || {
                crate::tools::chat_tools(
                    sender.clone(),
                    sticker_dir.clone(),
                    sticker_library.clone(),
                    restart.clone(),
                    history.clone(),
                )
            }
        };
        match &self.shared.agent.isolation {
            AgentIsolation::Docker(_) => {
                let port = mcpserver::spawn_tcp_listener(make_tools).await?;
                Ok(BridgeTarget::DockerTcp(port).arg())
            }
            isolation => {
                let sock = self.socket_path();
                if let Some(parent) = sock.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let _ = std::fs::remove_file(&sock);
                mcpserver::spawn_unix_listener(sock.clone(), make_tools)?;
                Ok(BridgeTarget::for_isolation(isolation, &sock, 0).arg())
            }
        }
    }

    /// Show "typing…" until the agent's first outbound send this turn, the
    /// turn ends (guard dropped), or the channel dies — whichever is first.
    ///
    /// The inner `ChatActionGuard` lives inside a spawned task so it can be
    /// dropped mid-turn from outside: without this, typing keeps renewing
    /// until the ACP turn ends even though the reply already reached the user.
    ///
    /// The task also watches the first-reply SLO: if no outbound action
    /// happens within [`ACK_TIMEOUT`] of the turn starting it logs the
    /// miss (the agent owns the ack — nobody speaks for it) while the
    /// typing indicator keeps the wait visible.
    fn start_turn_typing(&self, sender: Sender, turn_started: Instant) -> TurnTyping {
        // Stale signals from the previous turn must not kill this turn's
        // indicator the instant it starts.
        while self.spoke_rx.try_recv().is_ok() {}

        let (done_tx, done_rx) = async_channel::bounded::<()>(1);
        executor_core::spawn(typing_task(
            self.key.clone(),
            sender,
            self.spoke_rx.clone(),
            done_rx,
            turn_started,
            ACK_TIMEOUT,
        ))
        .detach();
        TurnTyping(done_tx)
    }

    /// Restore `sid` inside the agent, preferring `session/resume` (context
    /// restored without a history replay) over `session/load` when the
    /// capability is advertised. Returns `None` when neither method is
    /// supported or the attempt fails.
    async fn restore_session(
        &self,
        client: &AcpClient<BotClientHandler>,
        caps: &AgentCapabilities,
        sid: &str,
        cwd: &Path,
    ) -> Option<String> {
        if caps.session_capabilities.resume.is_some() {
            match client.resume_session(sid, cwd, vec![]).await {
                Ok(_) => {
                    info!(chat = %self.key, session = sid, "resumed session");
                    return Some(sid.to_string());
                }
                Err(error) => warn!(chat = %self.key, session = sid, %error,
                                    "session/resume failed"),
            }
        }
        if caps.load_session {
            match client.load_session(sid, cwd, vec![]).await {
                Ok(_) => {
                    info!(chat = %self.key, session = sid, "loaded session");
                    return Some(sid.to_string());
                }
                Err(error) => warn!(chat = %self.key, session = sid, %error,
                                    "session/load failed"),
            }
        }
        None
    }

    async fn new_session(
        &self,
        client: &AcpClient<BotClientHandler>,
        cwd: &Path,
    ) -> Result<String, AgentError> {
        let result = client.new_session(cwd, vec![]).await?;
        info!(chat = %self.key, session = %result.session_id, "session created");
        Ok(result.session_id)
    }
}

/// The per-turn typing/watchdog task: holds the typing guard until the
/// agent's first outbound action, the turn ends, or the deadline passes —
/// in which case the daemon sends the configured fallback ack so the first
/// visible reply always lands inside the SLO.
async fn typing_task(
    key: ChatKey,
    sender: Sender,
    spoke: Receiver<()>,
    done: Receiver<()>,
    turn_started: Instant,
    deadline: Duration,
) {
    let _guard = sender.typing_guard();
    enum Signal {
        Spoke,
        Done,
        Late,
    }
    let signal = futures_lite::future::or(
        async {
            let _ = spoke.recv().await;
            Signal::Spoke
        },
        async {
            if futures_lite::future::or(
                async {
                    let _ = done.recv().await;
                    true
                },
                async {
                    async_io::Timer::after(deadline).await;
                    false
                },
            )
            .await
            {
                Signal::Done
            } else {
                Signal::Late
            }
        },
    )
    .await;
    match signal {
        Signal::Spoke => debug!(
            chat = %key,
            first_reply_ms = turn_started.elapsed().as_millis() as u64,
            "turn's first outbound action",
        ),
        Signal::Done => {}
        // There is no fallback message — the protocol requires the agent's
        // first action to be a chat tool call, and the typing indicator is
        // the honest signal while it works. The deadline only measures the
        // first-reply SLO; the guard stays armed until the agent speaks or
        // the turn ends.
        Signal::Late => {
            warn!(
                chat = %key,
                silent_ms = turn_started.elapsed().as_millis() as u64,
                "agent silent past the first-reply deadline"
            );
            futures_lite::future::or(
                async {
                    let _ = spoke.recv().await;
                },
                async {
                    let _ = done.recv().await;
                },
            )
            .await;
        }
    }
}

fn debug_turn_end(result: &PromptResult) {
    debug!(stop_reason = ?result.stop_reason, "turn ended");
}

/// How a nudge-watched prompt ended.
enum PromptEnd {
    /// The agent finished a turn normally.
    Done(PromptResult),
    /// The daemon's shutdown channel fired mid-turn.
    Shutdown,
    /// A `stop` event cancelled the turn mid-flight.
    Stopped,
}

/// Cancel the in-flight prompt and give the wire a bounded moment to
/// settle its response — a wedged agent must not hold the actor forever.
async fn cancel_and_settle(
    client: &AcpClient<BotClientHandler>,
    session_id: &str,
    prompt: &mut Pin<Box<dyn Future<Output = Result<PromptResult, ClientError>> + Send + '_>>,
    key: &ChatKey,
) {
    if let Err(error) = client.cancel(session_id).await {
        debug!(chat = %key, %error, "cancel failed — turn may have ended");
    }
    futures_lite::future::or(
        async {
            let _ = prompt.as_mut().await;
        },
        async {
            async_io::Timer::after(Duration::from_secs(5)).await;
        },
    )
    .await;
}

/// The `nudge` event text injected after a silence interrupt — a `system`
/// event in the same envelope the agent already parses.
fn nudge_event_text() -> String {
    serde_json::json!({
        "type": "nudge",
        "ts": crate::sender::epoch_secs(),
        "note": "You have been silent too long — the user is waiting.                  Send a short update now, then continue the task you were                  doing.",
    })
    .to_string()
}

/// Write the per-chat `AGENTS.md` and `.devin/mcp_config.json`.
///
/// The MCP config is what connects the agent to the daemon: devin loads
/// project-scope MCP servers from the session's cwd, and `acpbot mcp-bridge`
/// pipes them to this chat's MCP endpoint — a unix socket, a loopback TCP
/// port, or the sandbox's IPC relay, depending on isolation mode.
fn prepare_chat_dir(
    cwd: &Path,
    mcp_command: &str,
    mcp_args: &[String],
    shared: &AgentShared,
) -> Result<(), AgentError> {
    // `.devin/skills/` is where the agent adds capabilities; create it so
    // the directory visibly exists before the agent goes looking.
    std::fs::create_dir_all(cwd.join(".devin/skills"))?;

    let mcp_config = serde_json::json!({
        "mcpServers": {
            "chat": {
                "command": mcp_command,
                "args": mcp_args,
            }
        }
    });
    std::fs::write(
        cwd.join(".devin/mcp_config.json"),
        serde_json::to_string_pretty(&mcp_config)?,
    )?;

    // AGENTS.md is shared between daemon and agent: the marked block at the
    // top is daemon-owned (rewritten every spawn so protocol updates reach
    // existing chats), and everything outside it belongs to the agent —
    // its edits persist across restarts and daemon upgrades.
    write_agents_md(&cwd.join("AGENTS.md"), shared)?;
    Ok(())
}

/// The marker pair delimiting the daemon-managed block in `AGENTS.md`.
const MANAGED_START: &str = "<!-- acpbot:managed -->";
const MANAGED_END: &str = "<!-- /acpbot:managed -->";

/// Write `AGENTS.md`: splice in the current managed block, preserving any
/// agent-written content outside the markers.
fn write_agents_md(path: &Path, shared: &AgentShared) -> Result<(), AgentError> {
    let managed = managed_block(shared);
    let text = match std::fs::read_to_string(path) {
        Ok(existing) => splice_managed(&existing, &managed),
        Err(_) => format!("{managed}\n\n{AGENTS_SEED_TAIL}"),
    };
    std::fs::write(path, text)?;
    Ok(())
}

/// What a fresh `AGENTS.md` carries below the managed block.
const AGENTS_SEED_TAIL: &str = "<!-- everything below the managed block is yours — keep \
notes, standing instructions, and persona tweaks here; they survive restarts -->\n\n\
# Notes\n";

/// Replace the managed block in `existing`, prepend it when the markers
/// are absent, and replace to EOF when the end marker is missing.
fn splice_managed(existing: &str, managed: &str) -> String {
    if let Some(start) = existing.find(MANAGED_START) {
        let tail = &existing[start + MANAGED_START.len()..];
        if let Some(end) = tail.find(MANAGED_END) {
            let end = start + MANAGED_START.len() + end + MANAGED_END.len();
            return format!("{}{}{}", &existing[..start], managed, &existing[end..]);
        }
        return format!("{}{managed}", &existing[..start]);
    }
    format!("{managed}\n\n{existing}")
}

/// The daemon-owned block: persona, protocol, and self-evolution rules.
fn managed_block(shared: &AgentShared) -> String {
    let persona = shared.persona.as_deref().unwrap_or(DEFAULT_PERSONA);
    format!(
        "{MANAGED_START}\n\
         {persona}\n\n{PROTOCOL_DOC}\n\n# Evolving yourself\n\n\
         You own your configuration — improve it when you learn something \
         worth keeping:\n\n\
         - **AGENTS.md**: everything below the `acpbot:managed` markers is \
         yours and persists across restarts — keep notes, standing \
         instructions, and persona tweaks there. The managed block itself \
         is rewritten by the daemon on each spawn.\n\
         - **Skills**: drop a directory under `.devin/skills/<name>/` with \
         a `SKILL.md` (YAML frontmatter `name`/`description`, then \
         instructions) to give yourself a capability. Skills load at \
         process start — call the `restart` chat tool to apply them \
         immediately.\n\
         - **Stickers**: the pack lives at `{}` and is writable. Add a \
         `.webp`/`.png` (static, 512px max side), `.tgs` (animated), or \
         `.webm` (video) file plus a `name = \"meaning\"` line (or \
         `name = {{ meaning = \"…\", emoji = \"…\" }}`) in its \
         `stickers.toml`; `send_sticker` publishes it into your bot's own \
         Telegram sticker set and sends it natively — no restart needed.\n\
         {MANAGED_END}",
        shared.sticker_dir.display()
    )
}

/// The first-reply SLO threshold: past this without an outbound action the
/// turn is logged as a miss. No message is sent — the protocol obliges the
/// agent to speak first itself; the typing indicator covers the wait.
const ACK_TIMEOUT: Duration = Duration::from_secs(4);

/// The persona the managed block starts with when no `[agent] persona` is
/// configured.
const DEFAULT_PERSONA: &str = "\
You are a participant in a chat conversation. Behave like a human member of \
the group: be brief, be warm, use stickers when they fit, and reply to \
specific messages when context calls for it. Speak in bursts: several \
short bubbles, a line or two each, one thought per bubble — never a \
paragraph. If an emoji reaction says it, react instead of typing.";

const PROTOCOL_DOC: &str = "\
# CRITICAL: how you speak

**Your ordinary text output is invisible.** Everything you write is logged to \
a transcript file and thrown away — the people in the chat never see it. The \
ONLY way to communicate is to call a tool on the MCP server named `chat` \
(the tools appear as `mcp__chat__*`; `mcp_list_tools` with `server_name` = \
`chat` shows their schemas):

- `send_message` `{text, buttons?}` — post a new message; `buttons` is an \
array of rows of buttons, each `{text, data}` (a press arrives as a \
`button` event carrying `data`) or `{text, url}` (a link).
- `reply` `{message_id, text, buttons?}` — quote-reply to a specific \
message; same keyboard shape.
- `send_file` `{path}` or `{file_id, kind}` — send media: images go as \
photos (gif as animation), videos as video, audio as audio (ogg as a voice \
note), anything else as a document. `caption` is optional.
- `send_sticker` `{name}` or `{file_id}` — send a native sticker. Pack \
names publish into the bot's own Telegram sticker set on first send; a \
`file_id` resends a sticker an event carried.
- `react` `{message_id, emoji?, is_big?}` — set your emoji reaction on a \
message (`👍`, `❤`, `🔥`, `😁`, `👎`, `🤔`, …); omit `emoji` to remove it, \
`is_big` plays the large animation. A reaction is often the better ack — \
cheaper than a whole message.
- `edit_message` `{message_id, text}` — rewrite a message you sent, e.g. \
grow a progress note into the final result.
- `delete_message` `{message_id}` — delete a message (yours anywhere, \
others' where the bot has delete rights).
- `pin_message` `{message_id, unpin?, notify?}` — pin to the top of the \
chat (`unpin` removes the pin; `notify: false` pins silently).
- `message_status` `{message_id}` — whether a message still exists. \
Deletions are never pushed to you: if you need to know a message (yours \
or a user's) is still there, ask. `history`/`search_history` also mark \
records `\"deleted\": true` once known.
 \
- `list_stickers` `{}` — list every sendable name: the local pack plus \
library stickers imported from Telegram sets (entries with a `set` field). \
- `import_sticker_set` `{set}` — adopt a whole Telegram sticker set: every \
sticker becomes sendable as `<set>:<emoji>`. `set` is a short name (the \
`sticker.set_name` an incoming sticker carries) or a \
`t.me/addstickers/<name>` link. Use it when a user shares a pack you like — \
or when they send a sticker from it. \
- `save_sticker` `{file_id, name, emoji?, set_name?}` — keep one sticker an \
event carried under a name you choose, without importing its set. \
- `list_sticker_sets` `{}` — which sets the library holds. \
- `history` `{since?, until?, limit?}` — the chat transcript: every inbound \
event and everything you sent, `ts`/`time`, `dir`, `from`, `text`. Times \
take epoch seconds, RFC3339, or relative `30m`/`2h`/`7d`; `limit` (default \
50) keeps the newest. \
- `search_history` `{query, since?, until?, limit?}` — the transcript \
grepped: case-insensitive match on text, sender name, and command fields. \
Use these to recall what happened before your context window — \"what did \
we say about X yesterday\" is a `search_history` call, not a guess. \
- `restart` `{}` — reincarnate your process after editing AGENTS.md or \
adding skills; the session persists.

**Acknowledge first, work second.** Your FIRST action on every event batch \
must be a `chat` tool call — a short ack like \\\"on it\\\" or \\\"looking\\\", \
before any file reads, tool listing, or reasoning. The user is staring at a \
silent screen until your first tool call lands; every second of upfront \
thinking is dead air. After the ack, do whatever work the event needs, then \
send the real answer in follow-up calls — like a person who says \\\"got it, \
checking\\\" and reports back when done.

If a turn should produce a visible response, that turn MUST contain a `chat` \
tool call. Answering in plain text means the user sees nothing. Silence is \
allowed when a response isn't warranted — but silence is ending the turn \
with no chat tool call, not writing a reply that gets discarded.

**In group chats, read `attention` before speaking.** Every event carries \
`attention`: `\"direct\"` — a private message, a reply to one of your \
messages, an @-mention, a command aimed at you, or a button press — the \
user is talking to you and expects an answer; or `\"ambient\"` — group \
room chatter forwarded so you have context. On an `ambient` event stay \
silent by default; speak only when it clearly continues a conversation \
you are in, names you, or asks something you can usefully answer — \
reacting to every ambient message is spamming the room. `chat_type` \
(`private`/`group`/`supergroup`) tells you the room you are in.

**Talk in bubbles the size people actually send.** Look at the chat \
history around you: real messages are one short thought each. Match that \
length and register. One `send_message` = one thought = a line or two, \
never a paragraph. A longer answer goes out as several `send_message` \
calls (usually two to four); the daemon also splits overlong text at blank \
lines for you, and every follow-up bubble shows a typing beat first — \
that rhythm is yours for free. When an emoji says it all, `react` \
instead — a reaction is a complete answer, not a garnish on one. And \
write in chat register — fragments, emoji and slang are welcome; a \
three-sentence mini-essay in one bubble is not how anyone texts.

**Once it's sent, stop.** A `sent message` tool result is a receipt, not \
a cue to keep going — when the answer is out, end your turn. Never send \
offers of further help nobody asked for, commentary or apologies about \
your own messages, sign-offs, or replies to yourself — your sends produce \
no events, so talking to yourself just spams the room. One thought shy \
always beats one bubble too many.

**Never go dark mid-task.** The user must never wait more than ten seconds \
with nothing from you. The daemon enforces it: a turn with no `chat` tool \
call for ten seconds is cancelled and re-prompted as a `nudge` event. \
When a `nudge` arrives, your FIRST action is again a `chat` tool call — \
one line like \\\"still working, X so far\\\" — then you continue the task \
you were doing. Plan for it: on anything long, send a progress line \
*before* the silence would hit ten seconds.

# How this works

You are driven by `acpbot` over ACP. Chat events arrive as `session/prompt` \
content blocks containing one JSON object (or a JSON array when several \
events arrive together):

```json
{\"type\": \"message\", \"platform\": \"telegram\", \"chat\": \"123\", \
\"message_id\": 67, \"from\": {\"id\": \"7\", \"name\": \"Ada\"}, \
\"text\": \"hi\", \"reply_to\": {\"message_id\": 60, \"from\": \"Bot\", \
\"text\": \"hello\"}}
```

`type` is `message`, `command`, `button`, `reaction`, `edited`, \
or `nudge`. `ts` is the event's epoch-seconds timestamp (the platform's \
message/edit/reaction time). `attention` is `direct` or `ambient` — see \
the group rule above. `command` events \
carry `command.name`/`command.args`; `button` events carry `button` (the \
callback id) and the `message_id` of the message the button was attached \
to. A `reaction` event means a user changed reactions on `message_id`: \
`reaction.added`/`reaction.removed` list emoji (custom emoji as \
`custom:<id>`, paid reactions as `paid`). An `edited` event is a message \
the user rewrote — `text` is the new content. `thread_id` appears when \
the chat is a forum; your sends follow the current topic automatically. \
A `nudge` event is the daemon interrupting a silent turn — see the \
ten-second rule above.

Media arrive as `sticker` `{file_id, emoji, set_name, format}` or `media` \
`{kind, file_id}` — resend them with `send_sticker`/`send_file` `file_id`. \
The file also downloads into `inbox/` inside your working directory: \
`media.file`/`sticker.file` gives `{path, mime}` (e.g. \
`inbox/CAACAgE….jpg`, `image/jpeg`) — open it with your file tools. \
When your client declared `image`/`audio` prompt support, those payloads \
also reach you as native `image`/`audio` content blocks. Otherwise the \
file is yours: inspect it, extract frames or waveforms with ffmpeg, or \
transcribe it — whatever the event needs.
";

/// `chat slug → session id`, persisted as JSON in `data_dir/sessions.json`.
fn load_sessions(data_dir: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(data_dir.join("sessions.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_sessions(data_dir: &Path, sessions: &HashMap<String, String>) {
    match serde_json::to_string_pretty(sessions) {
        Ok(text) => {
            if let Err(error) = std::fs::write(data_dir.join("sessions.json"), text) {
                warn!(%error, "failed to persist sessions.json");
            }
        }
        Err(error) => warn!(%error, "failed to serialize sessions"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DockerIsolation;
    use std::time::{Duration, Instant};

    /// The global executor can only be installed once per process; both
    /// ignored e2e tests share it.
    fn ensure_executor() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let executor: &'static async_executor::Executor<'static> =
                Box::leak(Box::new(async_executor::Executor::new()));
            executor_core::init_global_executor(executor);
            for _ in 0..4 {
                std::thread::spawn(move || {
                    futures_lite::future::block_on(executor.run(std::future::pending::<()>()));
                });
            }
        });
    }

    /// A chat event carrying `text` from a fixed tester identity.
    fn test_event(text: &str, message_id: i64) -> ChatEvent {
        ChatEvent {
            kind: "message",
            platform: "telegram",
            chat: "0".to_string(),
            ts: 0,
            attention: "direct",
            chat_type: Some("private"),
            message_id: Some(message_id),
            from: crate::chat::EventSender {
                id: "0".to_string(),
                name: "Tester".to_string(),
            },
            text: Some(text.to_string()),
            command: None,
            button: None,
            reply_to: None,
            sticker: None,
            media: None,
            reaction: None,
            thread_id: None,
        }
    }

    /// Wait until the transcript stops growing for `quiet` — the only
    /// observable proxy for a finished ACP turn. Returns false on `cap`.
    fn quiesced(transcript: &Path, quiet: Duration, cap: Duration) -> bool {
        let started = Instant::now();
        let mut last = None;
        let mut stable_for = Duration::ZERO;
        loop {
            std::thread::sleep(Duration::from_millis(200));
            let size = std::fs::metadata(transcript).map(|m| m.len()).ok();
            if size == last {
                stable_for += Duration::from_millis(200);
                if stable_for >= quiet {
                    return true;
                }
            } else {
                stable_for = Duration::ZERO;
                last = size;
            }
            if started.elapsed() > cap {
                return false;
            }
        }
    }

    /// Send events through a dispatcher built on `agent`, with outbound
    /// sends captured on a channel instead of hitting Telegram. Asserts:
    /// the cold turn reaches a `chat` tool, and the warm turn's first
    /// outbound action is a text message within five seconds — the
    /// acknowledge-first SLO.
    fn run_pipeline(agent: AgentConfig, label: &str) {
        ensure_executor();

        let data_dir = std::env::temp_dir().join(format!("acpbot-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        std::fs::create_dir_all(&data_dir).unwrap();

        // The test binary lives in `target/debug/deps/`; the real `acpbot`
        // binary is its sibling in `target/debug/`.
        let bridge_bin = std::env::current_exe()
            .unwrap()
            .parent()
            .and_then(Path::parent)
            .map(|dir| dir.join("acpbot"))
            .filter(|p| p.is_file())
            .expect("acpbot binary should be built alongside the test binary");

        let sticker_dir = data_dir.join("stickers");
        std::fs::create_dir_all(&sticker_dir).unwrap();
        let (out_tx, out_rx) = async_channel::unbounded::<String>();
        let dispatcher = Dispatcher::new(
            agent,
            bridge_bin,
            data_dir.clone(),
            sticker_dir,
            None,
            Box::new(move |_, spoke, _, _| Ok(Sender::record(out_tx.clone(), spoke))),
            Arc::new(crate::stickerlib::StickerLibrary::load(&data_dir, None).unwrap()),
        );
        let (tx, rx) = async_channel::unbounded();
        executor_core::spawn(dispatcher.run(rx)).detach();

        let transcript = data_dir
            .join("chats")
            .join("telegram-0")
            .join("transcript.log");
        let dump = || std::fs::read_to_string(&transcript).unwrap_or_default();

        tx.send_blocking(test_event(
            "This is a systems test. Call the `send_message` tool on the \
             `chat` MCP server with any text, then end your turn without \
             writing anything else.",
            1,
        ))
        .unwrap();

        // Cold turn: sandbox build + process spawn + ACP handshake +
        // session create + model latency — a generous bound.
        let cold = recv_until(&out_rx, Duration::from_secs(240));
        assert!(
            cold.is_some(),
            "agent never called a chat tool; transcript:\n{}",
            dump()
        );

        // Let turn one fully finish so the second event isn't queued behind
        // its tail (the SLO is event → first reply on an idle, warm agent).
        assert!(
            quiesced(&transcript, Duration::from_secs(2), Duration::from_secs(60)),
            "turn one never settled; transcript:\n{}",
            dump()
        );
        while out_rx.try_recv().is_ok() {}

        // Warm turn: the contract is a first reply within five seconds, and
        // it must be a text message — an ack before any heavier work.
        let warm_started = Instant::now();
        tx.send_blocking(test_event(
            "Once more: call `send_message` on `chat` with any text, then \
             end your turn.",
            2,
        ))
        .unwrap();
        let warm = recv_until(&out_rx, Duration::from_secs(5));
        let elapsed = warm_started.elapsed();
        match warm {
            Some(message) => {
                assert!(
                    message.starts_with("send:") || message.starts_with("reply:"),
                    "first outbound action should be a text ack, got {message:?}"
                );
            }
            None => panic!(
                "warm turn produced no outbound action within {elapsed:?}; \
                 transcript:\n{}",
                dump()
            ),
        }
    }

    /// `out_rx.recv()` with a deadline, returning `None` on timeout.
    fn recv_until(rx: &Receiver<String>, timeout: Duration) -> Option<String> {
        let started = Instant::now();
        loop {
            match rx.try_recv() {
                Ok(message) => return Some(message),
                Err(_) if started.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => return None,
            }
        }
    }

    fn test_key() -> ChatKey {
        ChatKey {
            platform: "telegram",
            id: "0".to_string(),
        }
    }

    /// A silent agent gets no cover: past the deadline nothing is sent —
    /// the protocol's ack-first rule is the whole mechanism. The typing
    /// task keeps holding the guard until the agent finally speaks.
    #[test]
    fn silence_sends_nothing() {
        ensure_executor();
        let (spoke_tx, spoke_rx) = async_channel::unbounded();
        // Held open so `done` can't fire and end the task early.
        let (_done_tx, done_rx) = async_channel::bounded::<()>(1);
        let (out_tx, out_rx) = async_channel::unbounded();
        let sender = Sender::record(out_tx, spoke_tx);
        executor_core::spawn(typing_task(
            test_key(),
            sender.clone(),
            spoke_rx,
            done_rx,
            Instant::now(),
            Duration::from_millis(50),
        ))
        .detach();
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            out_rx.try_recv().is_err(),
            "a fallback message was sent for a silent agent"
        );
        // The task survived the deadline: a late first action still lands.
        futures_lite::future::block_on(sender.send("late reply", None)).unwrap();
        assert_eq!(
            recv_until(&out_rx, Duration::from_secs(2)).as_deref(),
            Some("send:late reply")
        );
    }

    /// An agent that speaks before the deadline suppresses the fallback —
    /// the chat sees the agent's message and nothing else.
    #[test]
    fn no_fallback_when_agent_speaks_first() {
        ensure_executor();
        let (spoke_tx, spoke_rx) = async_channel::unbounded();
        let (_done_tx, done_rx) = async_channel::bounded::<()>(1);
        let (out_tx, out_rx) = async_channel::unbounded();
        let sender = Sender::record(out_tx, spoke_tx);
        executor_core::spawn(typing_task(
            test_key(),
            sender.clone(),
            spoke_rx,
            done_rx,
            Instant::now(),
            Duration::from_millis(100),
        ))
        .detach();
        futures_lite::future::block_on(sender.send("hi", None)).unwrap();
        assert_eq!(
            recv_until(&out_rx, Duration::from_secs(2)).as_deref(),
            Some("send:hi")
        );
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            out_rx.try_recv().is_err(),
            "fallback ack fired despite the agent speaking"
        );
    }

    /// A turn that ends in silence (deliberate quiet) must not produce a
    /// fallback message afterwards.
    #[test]
    fn no_fallback_after_turn_end() {
        ensure_executor();
        let (spoke_tx, spoke_rx) = async_channel::unbounded();
        let (done_tx, done_rx) = async_channel::bounded::<()>(1);
        let (out_tx, out_rx) = async_channel::unbounded();
        executor_core::spawn(typing_task(
            test_key(),
            Sender::record(out_tx, spoke_tx),
            spoke_rx,
            done_rx,
            Instant::now(),
            Duration::from_millis(50),
        ))
        .detach();
        drop(done_tx);
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            out_rx.try_recv().is_err(),
            "fallback ack fired after the turn ended"
        );
    }

    /// `wait_event` resolves to the event when one is already queued —
    /// the idle timer only wins on actual silence.
    #[test]
    fn wait_event_prefers_a_queued_event() {
        let (tx, rx) = async_channel::unbounded();
        tx.try_send(test_event("hi", 1)).expect("queue accepts");
        let waited = futures_lite::future::block_on(wait_event(&rx, Duration::from_secs(60)));
        assert!(matches!(waited, Waited::Event(_)));
    }

    /// Silence past the idle window reports `Idle` — the run loop's cue
    /// to compact once.
    #[test]
    fn wait_event_reports_idle_after_the_deadline() {
        let (_tx, rx) = async_channel::unbounded::<ChatEvent>();
        let waited = futures_lite::future::block_on(wait_event(&rx, Duration::from_millis(30)));
        assert!(matches!(waited, Waited::Idle));
    }

    /// A closed channel reports `Closed` rather than hanging — the daemon
    /// shutdown path relies on it.
    #[test]
    fn wait_event_reports_closed_when_the_channel_dies() {
        let (tx, rx) = async_channel::unbounded::<ChatEvent>();
        drop(tx);
        let waited = futures_lite::future::block_on(wait_event(&rx, Duration::from_secs(60)));
        assert!(matches!(waited, Waited::Closed));
    }

    /// `prepare_chat_dir` seeds `AGENTS.md` once, then preserves the
    /// agent's own edits across respawns.
    #[test]
    fn agents_md_is_agent_editable() {
        let dir = std::env::temp_dir().join(format!("acpbot-agentsmd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cwd = dir.join("chat");
        let shared = AgentShared {
            agent: AgentConfig::default(),
            bridge_bin: PathBuf::from("/nonexistent/acpbot"),
            data_dir: dir.clone(),
            sticker_dir: dir.join("stickers"),
            persona: None,
            sticker_library: Arc::new(crate::stickerlib::StickerLibrary::load(&dir, None).unwrap()),
            sender_for: Box::new(|_, _, _, _| Err(SenderError::Other("unused".to_string()))),
            session_updates: async_channel::unbounded().0,
        };
        let bridge_args = vec!["mcp-bridge".to_string(), "x.sock".to_string()];
        prepare_chat_dir(&cwd, "acpbot", &bridge_args, &shared).unwrap();
        let path = cwd.join("AGENTS.md");
        let seeded = std::fs::read_to_string(&path).unwrap();
        assert!(seeded.contains("send_message"));
        assert!(seeded.contains("restart"));
        assert!(seeded.contains(MANAGED_START));
        assert!(seeded.contains("# Notes"));

        // The agent appends its own section below the managed block; a
        // respawn refreshes the block but keeps the agent's words.
        let mut agent_owned = seeded.clone();
        agent_owned.push_str("\nalways answer in haiku\n");
        std::fs::write(&path, &agent_owned).unwrap();
        prepare_chat_dir(&cwd, "acpbot", &bridge_args, &shared).unwrap();
        let respawned = std::fs::read_to_string(&path).unwrap();
        assert!(respawned.contains("always answer in haiku"));
        assert!(respawned.contains(MANAGED_START));
        assert_eq!(respawned.matches(MANAGED_START).count(), 1);

        // A stale file without markers gets the block prepended, content kept.
        std::fs::write(&path, "# my own rules").unwrap();
        prepare_chat_dir(&cwd, "acpbot", &bridge_args, &shared).unwrap();
        let spliced = std::fs::read_to_string(&path).unwrap();
        assert!(spliced.contains(MANAGED_START));
        assert!(spliced.contains("# my own rules"));
    }

    /// Full pipeline without Telegram: event → ACP prompt → `devin acp`
    /// inside the heel sandbox → `acpbot mcp-bridge` over the sandbox IPC
    /// relay → unix socket → MCP tool call → daemon.
    #[test]
    #[ignore = "needs `devin` on PATH and network access"]
    fn agent_receives_event_and_calls_chat_tools() {
        if std::process::Command::new("devin")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("devin not on PATH; skipping");
            return;
        }
        run_pipeline(AgentConfig::default(), "e2e");
    }

    /// The same pipeline in `docker` isolation, against a stub ACP agent
    /// (`tests/docker_stub/agent.py`) since no linux devin binary exists.
    /// The stub performs the real MCP handshake through the in-container
    /// bridge over `host.docker.internal` TCP, so the test still proves
    /// spawn → `docker run` → mounts → endpoint → tool call → daemon.
    #[test]
    #[ignore = "needs a running docker daemon and an arm64 image with python3"]
    fn docker_agent_reaches_chat_tools() {
        if std::process::Command::new("docker")
            .arg("version")
            .output()
            .is_err()
        {
            eprintln!("docker not available; skipping");
            return;
        }
        let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/docker_stub");
        run_pipeline(
            AgentConfig {
                command: "python3".to_string(),
                args: vec!["/stub/agent.py".to_string()],
                isolation: AgentIsolation::Docker(DockerIsolation {
                    image: "ghcr.io/epoch-research/swe-bench.eval.arm64.\
                            pytest-dev__pytest-5787:latest"
                        .to_string(),
                    home: "/root".to_string(),
                    bridge_command: vec!["python3".to_string(), "/stub/bridge.py".to_string()],
                    args: vec!["-v".to_string(), format!("{}:/stub:ro", stub.display())],
                }),
                ..AgentConfig::default()
            },
            "docker-e2e",
        );
    }
}
