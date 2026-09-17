//! Shared agent lifecycle.
//!
//! One `devin acp` child process and ACP session serves every chat: a single
//! actor owns the global event queue, so the model sees one context window
//! across all conversations. The per-chat state that cannot be shared —
//! platform [`Sender`]s, IM records, forum-topic cells — lives in the
//! [`ChatRouter`], which also tracks the chat whose events triggered the
//! in-flight turn so tools default to answering it.
//!
//! Continuity across incarnations is a file, not a session: a closing actor
//! makes the agent write `CONTINUITY.md`, and a spawning actor opens a
//! *fresh* session and injects that summary — so the harness or model can
//! change on every restart. A session id in `sessions.json` is only a
//! "died before handoff" marker: the next spawn restores it just long
//! enough to extract the summary, then starts fresh anyway.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::error::{AgentError, SenderError};
use aither_acp::{
    AcpClient, AgentCapabilities, AudioContent, ClientError, ContentBlock, ImageContent,
    PromptCapabilities, PromptParams, PromptResult, SessionLoadParams, SessionNewParams,
    SessionResumeParams, SessionSetConfigOptionParams, SessionSetModeParams, TextContent,
};
use async_channel::{Receiver, Sender as ChanSender};
use base64::Engine as _;
use tracing::{debug, error, info, warn};

use crate::chat::{ChatEvent, ChatKey};
use crate::config::{AgentConfig, AgentIsolation};
use crate::handler::BotClientHandler;
use crate::history::{History, history_path};
use crate::mcpserver;
use crate::sandbox::{AgentRuntime, BridgeTarget};
use crate::sender::Sender;

/// Builds a platform [`Sender`] bound to one chat. The channel carries the
/// global "the agent spoke" signal shared by all senders; the `AtomicI64`
/// is the chat's forum-topic cell and the `AtomicU64` the last-action
/// timestamp (epoch ms) every sender shares.
pub type SenderFactory = Box<
    dyn Fn(&ChatKey, ChanSender<()>, Arc<AtomicI64>, Arc<AtomicU64>) -> Result<Sender, SenderError>
        + Send
        + Sync,
>;

/// What the actor, router, and tools share: agent launch config, paths,
/// the sticker pack directory, and a factory that binds a platform
/// [`Sender`] to a chat.
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
    /// Persona text injected into the shared `AGENTS.md`.
    pub persona: Option<String>,
    /// Imported foreign sticker sets — bot-global, persisted under
    /// `data_dir`, shared by every chat's `send_sticker`.
    pub sticker_library: Arc<crate::stickerlib::StickerLibrary>,
    /// Builds a platform sender bound to a chat.
    pub sender_for: SenderFactory,
    /// Where the actor reports session lifecycle events; the dispatcher
    /// persists them so an unclean exit can be recovered on next run.
    pub session_updates: ChanSender<SessionUpdate>,
}

/// What the actor tells the dispatcher about the live session.
///
/// `sessions.json` no longer means "resume this on start" — a stored id is
/// an *awaiting handoff* marker: a session that was never given the chance
/// to write its `CONTINUITY.md` summary (crash, killed, timed-out handoff).
#[derive(Debug)]
pub enum SessionUpdate {
    /// A session was established; if the daemon dies before `Closed`, the
    /// next run must restore it long enough to extract the summary.
    Established(String),
    /// The session handed off cleanly — drop the stored id, nothing to
    /// recover.
    Closed,
}

/// Media above this many bytes stays an `inbox/` file path in the event JSON
/// rather than also being inlined as an `image`/`audio` content block —
/// keeps a huge attachment from dominating the prompt.
const MEDIA_INLINE_LIMIT: u64 = 8 * 1024 * 1024;

/// The `sessions.json` key — and `chats/` subdirectory — of the one shared
/// session. Per-chat slugs in a pre-shared `sessions.json` survive only so
/// the first spawn after an upgrade can recover their old session's
/// summary into the shared `CONTINUITY.md`.
const SHARED_SLUG: &str = "shared";

/// The file the outgoing session writes its handoff summary into — and the
/// file the next incarnation receives as its bootstrap prompt. Lives in
/// the shared working directory so it survives sessions and harness swaps.
const CONTINUITY_FILE: &str = "CONTINUITY.md";

/// Hard bound on a handoff or bootstrap prompt: a wedged agent must not
/// stall daemon shutdown or first-response forever — a timeout degrades
/// into the crash-recovery path or a memory-less start.
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(120);

/// The maintenance prompt that makes a session write its own handoff
/// note — plain text so any harness can follow it. The target file goes
/// in as an absolute path: a harness may resolve a bare `CONTINUITY.md`
/// against somewhere other than the session cwd (agy's file tools land
/// in its scratch dir, not `cwd`).
const HANDOFF_PROMPT: &str = "\
The daemon is closing this session permanently — shutdown, restart, or a \
harness swap. Write your handoff note — who you are, the chats and people \
you know, what you were in the middle of, and what the next incarnation \
needs to know — to the file at the exact absolute path below (create or \
overwrite it). Plain markdown, as compact as accuracy allows. Do not call \
any `chat` tools — nobody is watching; the file is this turn's only \
output. Write it, then end the turn.\n\nAbsolute path: ";

/// The preamble a fresh session's bootstrap prompt gets — the previous
/// incarnation's `CONTINUITY.md` text is appended after it.
const INJECT_PROMPT: &str = "\
You just came online — a fresh session after a daemon restart (the model \
or harness may have changed; this is how every incarnation starts). Below \
is CONTINUITY.md, the handoff note your previous self wrote before \
closing. Treat it as memory, not gospel: trust it for who you are and \
what was in flight, and verify anything load-bearing against the real \
chat records — `history`, `search_history`, `fetch_message`, `chat_info` \
hold every message the bot saw or sent and outlive any session. Absorb it \
quietly — no `chat` tool calls, nobody is waiting — then end the turn.";

/// The [`ChatKey`] an event belongs to.
fn key_of(event: &ChatEvent) -> ChatKey {
    ChatKey {
        platform: event.platform,
        id: event.chat.clone(),
    }
}

/// The registry from [`ChatKey`] to the per-chat state the shared agent
/// still needs: a chat's platform [`Sender`], its IM record [`History`],
/// and its forum-topic cell — plus `current`, the chat whose events
/// triggered the in-flight turn, which tools resolve to when their `chat`
/// argument is omitted.
///
/// Senders and histories are built lazily on first sight of a chat. The
/// mutex is held only for map operations, never across an `.await`.
pub struct ChatRouter {
    shared: Arc<AgentShared>,
    state: Mutex<RouterState>,
    /// Posted to by every sender on outbound output — capacity 1, so a
    /// full channel is the "the agent already spoke this turn" flag.
    spoke: ChanSender<()>,
    /// Epoch ms of the agent's last outbound action anywhere — the nudge
    /// watchdog measures user-visible silence against it.
    last_action: Arc<AtomicU64>,
    /// The platform the daemon serves — the tag a bare `chat` argument
    /// resolves to (a daemon only ever bridges one platform).
    platform: &'static str,
}

#[derive(Default)]
struct RouterState {
    /// One sender per chat, built on demand.
    senders: HashMap<ChatKey, Sender>,
    /// One IM record per chat.
    histories: HashMap<ChatKey, Arc<History>>,
    /// One forum-topic cell per chat — lives outside the senders so the
    /// actor can set a topic before that chat's sender exists.
    threads: HashMap<ChatKey, Arc<AtomicI64>>,
    /// The chat whose events triggered the in-flight turn.
    current: Option<ChatKey>,
}

impl RouterState {
    /// The IM record of `key` (`<data>/chats/<slug>/history.jsonl`),
    /// opened on first use.
    fn history(&mut self, key: &ChatKey, data_dir: &Path) -> Arc<History> {
        self.histories
            .entry(key.clone())
            .or_insert_with(|| {
                Arc::new(History::open(history_path(
                    &data_dir.join("chats").join(key.slug()),
                )))
            })
            .clone()
    }
}

impl ChatRouter {
    fn new(
        shared: Arc<AgentShared>,
        spoke: ChanSender<()>,
        last_action: Arc<AtomicU64>,
        platform: &'static str,
    ) -> Self {
        Self {
            shared,
            state: Mutex::new(RouterState::default()),
            spoke,
            last_action,
            platform,
        }
    }

    /// The agent config, paths, and sticker stores the tools also need.
    pub(crate) fn shared(&self) -> &AgentShared {
        &self.shared
    }

    /// The chat the in-flight turn answers — the tools' default `chat`.
    pub fn current(&self) -> Option<ChatKey> {
        self.state.lock().expect("chat router").current.clone()
    }

    /// Point tool defaults at the chat whose events triggered the turn.
    pub(crate) fn set_current(&self, key: ChatKey) {
        self.state.lock().expect("chat router").current = Some(key);
    }

    /// Record the forum topic `key`'s current events arrived in (`0` =
    /// none). The cell outlives any one sender, so this also reaches
    /// senders built later.
    pub(crate) fn set_thread(&self, key: &ChatKey, thread_id: i64) {
        self.state
            .lock()
            .expect("chat router")
            .threads
            .entry(key.clone())
            .or_default()
            .store(thread_id, Ordering::Relaxed);
    }

    /// The IM record of `key`, opened on first use.
    pub fn history(&self, key: &ChatKey) -> Arc<History> {
        self.state
            .lock()
            .expect("chat router")
            .history(key, &self.shared.data_dir)
    }

    /// The [`Sender`] bound to `key`, built through the platform factory
    /// on first sight of the chat.
    pub fn sender(&self, key: &ChatKey) -> Result<Sender, SenderError> {
        let mut state = self.state.lock().expect("chat router");
        if let Some(sender) = state.senders.get(key) {
            return Ok(sender.clone());
        }
        let thread = state.threads.entry(key.clone()).or_default().clone();
        let history = state.history(key, &self.shared.data_dir);
        let sender =
            (self.shared.sender_for)(key, self.spoke.clone(), thread, self.last_action.clone())?
                .with_history(history);
        state.senders.insert(key.clone(), sender.clone());
        Ok(sender)
    }

    /// Resolve a tool's `chat` argument: `"platform:id"`, or a bare id on
    /// the daemon's platform. `None` resolves to the in-flight turn's chat.
    pub fn resolve(&self, chat: Option<&str>) -> Result<ChatKey, String> {
        match chat {
            Some(raw) => {
                let (platform, id) = raw
                    .split_once(':')
                    .map_or((self.platform, raw.trim()), |(p, id)| (p.trim(), id.trim()));
                if platform != self.platform {
                    return Err(format!(
                        "unknown platform {platform:?} — this bot serves {:?} only",
                        self.platform
                    ));
                }
                if id.is_empty() {
                    return Err(format!("missing chat id in {raw:?}"));
                }
                Ok(ChatKey {
                    platform: self.platform,
                    id: id.to_string(),
                })
            }
            None => self
                .current()
                .ok_or_else(|| "no active chat yet — pass `chat` explicitly".to_string()),
        }
    }

    /// `resolve` + `sender` — a tool call's outbound target.
    pub fn sender_for(&self, chat: Option<&str>) -> Result<Sender, String> {
        let key = self.resolve(chat)?;
        self.sender(&key).map_err(|e| e.to_string())
    }

    /// `resolve` + `history` + `sender` — the IM-record tools' target.
    /// The sender comes along so returned records can be probed for
    /// deletion.
    pub fn history_for(&self, chat: Option<&str>) -> Result<(Arc<History>, Sender), String> {
        let key = self.resolve(chat)?;
        let sender = self.sender(&key).map_err(|e| e.to_string())?;
        Ok((self.history(&key), sender))
    }
}

/// Forwards every chat's events to the one shared agent actor.
pub struct Dispatcher {
    shared: Arc<AgentShared>,
    /// The shared actor's queue and task — spawned on the first event,
    /// respawned if its run loop ever ends early.
    actor: Option<(ChanSender<ChatEvent>, executor_core::AnyExecutorTask<()>)>,
    /// `sessions.json` — slug → session id. The live session sits under
    /// [`SHARED_SLUG`]; per-chat slugs remain only for the upgrade handoff.
    known_sessions: HashMap<String, String>,
    /// Receives the session lifecycle events the actor reports.
    session_ids: Receiver<SessionUpdate>,
    /// Cloned into the actor; dropping the sender ends its in-flight turn
    /// so a daemon shutdown never leaves a sandboxed agent running.
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
            actor: None,
            known_sessions,
            session_ids,
            shutdown: (Some(shutdown_tx), shutdown_rx),
        }
    }

    /// Consume events forever, forwarding each to the shared actor.
    pub async fn run(mut self, events: Receiver<ChatEvent>) {
        loop {
            // `race`, not `or`: `or` keeps waiting on the other arm after an
            // `Err`, and `session_ids` never closes (its sender lives in
            // `self.shared`) — so a closed event channel must complete on
            // its own for shutdown to ever leave this loop.
            let event = futures_lite::future::race(
                async { events.recv().await.map(|e| Routed::Event(Box::new(e))) },
                async { self.session_ids.recv().await.map(Routed::SessionUpdate) },
            )
            .await;
            match event {
                Ok(Routed::Event(event)) => self.dispatch(*event).await,
                Ok(Routed::SessionUpdate(update)) => {
                    self.apply_session_update(update);
                }
                Err(_) => break,
            }
        }

        // The event channel closed: the daemon is shutting down. Closing the
        // shutdown channel interrupts an in-flight turn; dropping the actor's
        // sender ends its run loop; awaiting its task lets the actor write
        // the handoff summary and drop its runtime — a heel Sandbox kills
        // the agent process on the way out.
        drop(self.shutdown.0.take());
        if let Some((tx, task)) = self.actor.take() {
            drop(tx);
            task.await;
        }
        // The actor can report session updates right up to its task's end —
        // drain what arrived so recovery state lands on disk.
        while let Ok(update) = self.session_ids.try_recv() {
            self.apply_session_update(update);
        }
    }

    /// Persist one actor-reported session event into `sessions.json`: an
    /// established session becomes the recovery handle, a cleanly closed
    /// one drops the marker entirely.
    fn apply_session_update(&mut self, update: SessionUpdate) {
        match update {
            SessionUpdate::Established(sid) => {
                // The live session supersedes every other id — it is the
                // one a crash recovery would have to summarize.
                self.known_sessions.clear();
                self.known_sessions.insert(SHARED_SLUG.to_string(), sid);
            }
            SessionUpdate::Closed => {
                self.known_sessions.remove(SHARED_SLUG);
            }
        }
        save_sessions(&self.shared.data_dir, &self.known_sessions);
    }

    async fn dispatch(&mut self, event: ChatEvent) {
        if let Some((tx, _)) = &self.actor
            && tx.send(event.clone()).await.is_ok()
        {
            return;
        }

        // No actor yet (or it died): spawn the shared one. A session id
        // under `shared` — or, on upgrade from the per-chat layout, the
        // triggering chat's slug — marks a session that never handed off;
        // the actor restores it only to extract CONTINUITY.md before
        // starting fresh.
        let key = key_of(&event);
        let session_id = self
            .known_sessions
            .get(SHARED_SLUG)
            .or_else(|| self.known_sessions.get(&key.slug()))
            .cloned();

        let (tx, rx) = async_channel::unbounded();
        // Capacity 1: a full channel is the "the agent already spoke
        // this turn" flag, and the drain at turn start resets it.
        let (spoke_tx, spoke_rx) = async_channel::bounded(1);
        let last_action = Arc::new(AtomicU64::new(crate::sender::epoch_ms()));
        let (restart_tx, restart_rx) = async_channel::unbounded();
        let actor = ChatActor {
            shared: self.shared.clone(),
            router: Arc::new(ChatRouter::new(
                self.shared.clone(),
                spoke_tx,
                last_action.clone(),
                key.platform,
            )),
            client: None,
            session_id,
            mcp_endpoint: None,
            runtime: None,
            child: None,
            prompt_caps: PromptCapabilities::default(),
            compact_supported: Arc::new(AtomicBool::new(false)),
            session_turns: 0,
            spoke_rx,
            last_action,
            restart_tx,
            restart_rx,
            shutdown_rx: self.shutdown.1.clone(),
        };
        let task = executor_core::spawn(actor.run(rx));
        self.actor = Some((tx.clone(), task));
        if tx.send(event).await.is_err() {
            error!("shared agent actor refused its first event");
        }
    }
}

enum Routed {
    Event(Box<ChatEvent>),
    SessionUpdate(SessionUpdate),
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

/// The one agent process, ACP session, and event queue every chat shares.
struct ChatActor {
    shared: Arc<AgentShared>,
    /// Per-chat senders, IM records, and topic cells — plus `current`,
    /// the chat whose events triggered the in-flight turn. Shared with
    /// every tool the chat MCP endpoint serves.
    router: Arc<ChatRouter>,
    client: Option<AcpClient<BotClientHandler>>,
    /// A session awaiting handoff — handed in by the dispatcher or held
    /// across a mid-run respawn. Restored only to extract `CONTINUITY.md`,
    /// then replaced by a fresh session; never resumed into duty.
    session_id: Option<String>,
    /// The bound MCP endpoint — socket path on unix, loopback TCP where
    /// unix sockets don't exist, docker's gateway TCP for `docker`.
    mcp_endpoint: Option<crate::mcpserver::ChatEndpoint>,
    /// The agent's isolation runtime (sandbox/container), once created.
    runtime: Option<AgentRuntime>,
    /// The sandboxed child handle of the current agent process, if any.
    child: Option<heel::Child>,
    /// Drained per turn to stop the typing indicator on the first send;
    /// every sender posts to its sender half.
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
    /// Turns the current session has run. `0` means a fresh session with
    /// nothing to summarize — a handoff then would only overwrite a better
    /// `CONTINUITY.md` with a thin one, so `graceful_close` skips it.
    session_turns: u64,
}

impl ChatActor {
    /// The shared agent working directory (`<data_dir>/chats/shared/`).
    fn cwd(&self) -> PathBuf {
        self.shared.data_dir.join("chats").join(SHARED_SLUG)
    }

    /// The IPC socket the MCP bridge connects to.
    fn socket_path(&self) -> PathBuf {
        self.shared
            .data_dir
            .join("run")
            .join(format!("{SHARED_SLUG}.sock"))
    }

    /// The `command`/`args` `mcp_config.json` should spawn to reach the
    /// chat MCP endpoint — the host `acpbot` binary for `none`/`native`,
    /// the configured in-container command for `docker`.
    fn mcp_server_entry(&self) -> Result<(String, Vec<String>), AgentError> {
        let endpoint = self.mcp_endpoint.as_ref().expect("bound first");
        let target = BridgeTarget::for_isolation(&self.shared.agent.isolation, endpoint).arg();
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

    /// The transcript file inside the shared cwd.
    fn transcript_path(&self) -> PathBuf {
        self.cwd().join("transcript.log")
    }

    /// The IM record of the chat `event` belongs to.
    fn log_event(&self, event: &ChatEvent) {
        self.router.history(&key_of(event)).append_event(event);
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
                        Err(_) => break,
                    }
                } else {
                    match wait_event(&rx, idle).await {
                        Waited::Event(event) => *event,
                        Waited::Closed => break,
                        Waited::Idle => {
                            if let Err(error) = self.compact().await {
                                warn!(%error, "idle compaction failed");
                            }
                            idle_compacted = true;
                            continue;
                        }
                    }
                };
                self.log_event(&first);
                batch.push(first);
            }
            idle_compacted = false;
            // Coalesce events that arrived while a turn was in flight —
            // a batch can mix chats, and each event lands in its own
            // chat's IM record.
            while let Ok(event) = rx.try_recv() {
                self.log_event(&event);
                batch.push(event);
            }
            if let Err(error) = self.turn(&batch, &rx, &mut pending).await {
                error!(%error, "turn failed");
            }
        }

        // The event stream ended — the daemon is going down. Give the live
        // session its one chance to write CONTINUITY.md before the process
        // dies; a written file releases the session id, a failed one leaves
        // it for the next run's recovery.
        self.graceful_close().await;
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
            debug!("idle; agent has no compact command, skipping");
            return Ok(());
        }
        info!("idle; compacting session");
        let result = client
            .prompt(PromptParams::new(
                session_id.clone(),
                vec![ContentBlock::Text(TextContent {
                    text: "/compact".to_string(),
                    annotations: None,
                    meta: None,
                })],
            ))
            .await?;
        debug!(stop = ?result.stop_reason, "idle compaction done");
        Ok(())
    }

    /// Deliver one batch of events to the agent as a single prompt.
    ///
    /// A batch can mix chats when events coalesced mid-turn; the freshest
    /// event's chat becomes the turn's `current` — the default target for
    /// the agent's tool calls.
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
        let current = key_of(batch.last().expect("a batch is never empty"));
        self.router.set_current(current.clone());

        // Forum topics stay per chat: each chat present in the batch gets
        // the last `thread_id` its events carried (`0` clears).
        let mut threads: HashMap<ChatKey, i64> = HashMap::new();
        for event in batch {
            threads.insert(key_of(event), event.thread_id.unwrap_or(0));
        }
        for (key, thread_id) in threads {
            self.router.set_thread(&key, thread_id);
        }

        self.ensure_ready().await?;
        self.session_turns += 1;

        // Pull every file the events carry into `inbox/` so the agent can
        // open the bytes; image payloads also go into the prompt as `image`
        // content blocks so vision-capable models see them directly. The
        // download goes through the origin chat's sender (platform APIs
        // are chat-bound) into the shared working directory.
        let agent_dir = self.cwd();
        let mut batch = batch.to_vec();
        for event in &mut batch {
            let sender = self.router.sender(&key_of(event))?;
            if let Some(media) = &mut event.media {
                media.file = sender.fetch_media(&media.file_id, &agent_dir).await?;
            }
            if let Some(sticker) = &mut event.sticker {
                sticker.file = sender.fetch_media(&sticker.file_id, &agent_dir).await?;
            }
            // A quoted message's attachments matter just as much — a user
            // answering "who is this?" about a sticker needs its bytes.
            if let Some(reply) = &mut event.reply_to {
                if let Some(media) = &mut reply.media {
                    media.file = sender.fetch_media(&media.file_id, &agent_dir).await?;
                }
                if let Some(sticker) = &mut reply.sticker {
                    sticker.file = sender.fetch_media(&sticker.file_id, &agent_dir).await?;
                }
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
            meta: None,
        })];
        for file in batch.iter().flat_map(ChatEvent::files) {
            let is_image = self.prompt_caps.image && file.mime.starts_with("image/");
            let is_audio = self.prompt_caps.audio && file.mime.starts_with("audio/");
            if !is_image && !is_audio {
                continue;
            }
            let bytes = async_fs::read(&file.path).await?;
            if bytes.len() as u64 > MEDIA_INLINE_LIMIT {
                continue;
            }
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            let mime_type = file.mime.clone();
            prompt.push(if is_image {
                ContentBlock::Image(ImageContent {
                    data: Some(data),
                    uri: None,
                    mime_type,
                    annotations: None,
                    meta: None,
                })
            } else {
                ContentBlock::Audio(AudioContent {
                    data,
                    mime_type,
                    annotations: None,
                    meta: None,
                })
            });
        }

        let _typing = self.start_turn_typing(current, turn_started);
        self.last_action
            .store(crate::sender::epoch_ms(), Ordering::Relaxed);

        let client = self.client.clone().expect("ensure_ready ran");
        let session_id = self.session_id.clone().expect("ensure_ready ran");

        let outcome = match self
            .prompt_with_nudges(&client, &session_id, prompt.clone(), rx, pending)
            .await
        {
            // A daemon shutdown interrupts the prompt wait — but the agent
            // still owes its CONTINUITY.md. Keep client, child, and runtime
            // alive; the run loop's graceful_close runs the handoff and
            // owns the teardown.
            Ok(PromptEnd::Shutdown) => return Ok(()),
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
                warn!("agent process died; respawning");
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
        // respawns, and the carried-over session id triggers the recovery
        // handoff (write CONTINUITY.md, then a fresh session + inject).
        // Drain every queued request so duplicate calls don't cause
        // repeated restarts.
        let mut restart = false;
        while self.restart_rx.try_recv().is_ok() {
            restart = true;
        }
        if restart {
            info!("agent requested restart");
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
        > = Box::pin(client.prompt(PromptParams::new(session_id, content)));
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
                // Cancel before yielding the session: the shutdown handoff
                // prompt can't run while this turn is still in flight.
                Race::Shutdown => {
                    cancel_and_settle(client, session_id, &mut prompt).await;
                    return Ok(PromptEnd::Shutdown);
                }
                Race::EventsClosed => events_open = false,
                Race::Event(event) => {
                    self.log_event(&event);
                    // A `stop` only cancels the turn it belongs to: a stop
                    // from another chat is that chat's next-turn business,
                    // not a veto over this one's work.
                    let owns_turn = self
                        .router
                        .current()
                        .is_some_and(|current| current == key_of(&event));
                    if event.is_stop() && owns_turn {
                        info!(chat = %key_of(&event), "user said stop; cancelling turn");
                        cancel_and_settle(client, session_id, &mut prompt).await;
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
                        chat = ?self.router.current(),
                        silent_ms, "agent silent past the nudge deadline; interrupting"
                    );
                    cancel_and_settle(client, session_id, &mut prompt).await;
                    // The nudged turn gets a fresh silence budget — it must
                    // still speak within `nudge_after` or be interrupted
                    // again.
                    self.last_action
                        .store(crate::sender::epoch_ms(), Ordering::Relaxed);
                    prompt = Box::pin(client.prompt(PromptParams::new(
                        session_id,
                        vec![ContentBlock::Text(TextContent {
                            text: nudge_event_text(self.router.current().as_ref()),
                            annotations: None,
                            meta: None,
                        })],
                    )));
                }
            }
        }
    }

    /// Bring up the chat MCP endpoint, the isolation runtime, the agent
    /// process, and the ACP session.
    async fn ensure_ready(&mut self) -> Result<(), AgentError> {
        if self.mcp_endpoint.is_none() {
            self.mcp_endpoint = Some(self.bind_mcp_endpoint().await?);
        }

        let cwd = self.cwd();
        std::fs::create_dir_all(&cwd)?;

        if self.runtime.is_none() {
            self.runtime = Some(
                AgentRuntime::create(
                    &self.shared.agent.isolation,
                    &self.shared.agent,
                    &cwd,
                    self.mcp_endpoint.as_ref().expect("bound above"),
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
            agent = init.agent_info.as_ref().map_or("?", |i| i.name.as_str()),
            image = self.prompt_caps.image,
            audio = self.prompt_caps.audio,
            "agent initialized"
        );

        // A carried-over session id means the previous incarnation never
        // handed off — a crash, a kill, or a handoff that ran out of time.
        // Restore it only long enough to extract `CONTINUITY.md`, then let
        // it go: this daemon never resumes a session into active duty, so
        // the harness or model is free to change between incarnations.
        if let Some(old) = self.session_id.take()
            && let Some(sid) = self
                .restore_session(&client, &init.agent_capabilities, &old, &cwd)
                .await
        {
            if self.write_continuity(&client, &sid).await {
                info!(session = sid, "recovered continuity from previous session");
                let _ = self.shared.session_updates.try_send(SessionUpdate::Closed);
            } else {
                warn!(
                    session = sid,
                    "continuity recovery failed; starting fresh anyway"
                );
            }
        }

        let session_id = self.new_session(&client, &cwd).await?;

        if let Err(error) = client
            .set_mode(SessionSetModeParams::new(
                session_id.clone(),
                self.shared.agent.mode.clone(),
            ))
            .await
        {
            warn!(%error, "set_mode failed");
        }
        if let Err(error) = client
            .set_config_option(SessionSetConfigOptionParams::new(
                session_id.clone(),
                "model",
                self.shared.agent.model.clone(),
            ))
            .await
        {
            warn!(%error, "set_config_option(model) failed");
        }

        self.session_id = Some(session_id.clone());
        self.session_turns = 0;
        let _ = self
            .shared
            .session_updates
            .try_send(SessionUpdate::Established(session_id.clone()));
        // Bootstrap memory goes in as the first turn — a plain prompt, so
        // the same file carries across harnesses and models.
        self.inject_continuity(&client, &session_id).await;
        self.client = Some(client);
        Ok(())
    }

    /// Bind the shared MCP endpoint — a unix socket on unix, loopback TCP
    /// on Windows, docker's gateway listener for `docker` — and return it.
    /// Every accepted connection gets tools backed by the same router.
    async fn bind_mcp_endpoint(&mut self) -> Result<crate::mcpserver::ChatEndpoint, AgentError> {
        let make_tools = {
            let router = self.router.clone();
            let restart = self.restart_tx.clone();
            move || crate::tools::chat_tools(router.clone(), restart.clone())
        };
        match &self.shared.agent.isolation {
            AgentIsolation::Docker(_) => Ok(mcpserver::bind_docker_endpoint(make_tools).await?),
            _ => {
                let sock = self.socket_path();
                if let Some(parent) = sock.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let _ = std::fs::remove_file(&sock);
                Ok(mcpserver::bind_host_endpoint(sock, make_tools).await?)
            }
        }
    }

    /// Show "typing…" in `key`'s chat until the agent's first outbound send
    /// this turn, the turn ends (guard dropped), or the channel dies —
    /// whichever is first.
    ///
    /// The inner `ChatActionGuard` lives inside a spawned task so it can be
    /// dropped mid-turn from outside: without this, typing keeps renewing
    /// until the ACP turn ends even though the reply already reached the user.
    ///
    /// The task also watches the first-reply SLO: if no outbound action
    /// happens within [`ACK_TIMEOUT`] of the turn starting it logs the
    /// miss (the agent owns the ack — nobody speaks for it) while the
    /// typing indicator keeps the wait visible.
    fn start_turn_typing(&self, key: ChatKey, turn_started: Instant) -> TurnTyping {
        // Stale signals from the previous turn must not kill this turn's
        // indicator the instant it starts.
        while self.spoke_rx.try_recv().is_ok() {}

        let sender = match self.router.sender(&key) {
            Ok(sender) => sender,
            Err(error) => {
                warn!(chat = %key, %error, "typing indicator unavailable");
                return TurnTyping(async_channel::bounded::<()>(1).0);
            }
        };
        let (done_tx, done_rx) = async_channel::bounded::<()>(1);
        executor_core::spawn(typing_task(
            key,
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
            match client
                .resume_session(SessionResumeParams::new(sid, cwd))
                .await
            {
                Ok(_) => {
                    info!(session = sid, "resumed session");
                    return Some(sid.to_string());
                }
                Err(error) => warn!(session = sid, %error,
                                    "session/resume failed"),
            }
        }
        if caps.load_session {
            match client.load_session(SessionLoadParams::new(sid, cwd)).await {
                Ok(_) => {
                    info!(session = sid, "loaded session");
                    return Some(sid.to_string());
                }
                Err(error) => warn!(session = sid, %error,
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
        let result = client.new_session(SessionNewParams::new(cwd)).await?;
        info!(session = %result.session_id, "session created");
        Ok(result.session_id)
    }

    /// End the actor: give the live session its one chance to write
    /// `CONTINUITY.md`, then drop client, process, and runtime. A written
    /// file means the handoff completed — the session id is reported
    /// `Closed` so the next spawn starts fresh; a failed handoff leaves
    /// the id recorded, so the next spawn's recovery retries it.
    async fn graceful_close(&mut self) {
        let handed_off = if self.session_id.is_none() {
            false
        } else if self.session_turns == 0 {
            // A session that never ran a turn has nothing to summarize —
            // and asking it anyway would overwrite a better CONTINUITY.md
            // (one just recovered into place) with an empty incarnation's
            // note. It still counts as closed: nothing exists to recover.
            true
        } else if let (Some(client), Some(sid)) = (self.client.clone(), self.session_id.clone()) {
            self.write_continuity(&client, &sid).await
        } else {
            false
        };
        if handed_off {
            info!(session = ?self.session_id, "session closed");
            self.session_id = None;
            let _ = self.shared.session_updates.try_send(SessionUpdate::Closed);
        }
        self.client = None;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }
        self.runtime = None;
    }

    /// Ask the session to leave its handoff note: `/compact` first when
    /// the harness advertises it — the post-compact context is already the
    /// distilled state the note should carry — then a prompt that writes
    /// `CONTINUITY.md`. `true` only when the file was (re)written by this
    /// call; anything less keeps the session resumable for a retry.
    async fn write_continuity(
        &self,
        client: &AcpClient<BotClientHandler>,
        session_id: &str,
    ) -> bool {
        let path = self.cwd().join(CONTINUITY_FILE);
        let before = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        if self.compact_supported.load(Ordering::Relaxed)
            && let Err(error) = self
                .maintenance_prompt(client, session_id, "/compact".to_string())
                .await
        {
            warn!(%error, "handoff compaction failed");
        }
        if let Err(error) = self
            .maintenance_prompt(
                client,
                session_id,
                format!("{HANDOFF_PROMPT}{}", path.display()),
            )
            .await
        {
            warn!(%error, "handoff prompt failed");
            return false;
        }
        let wrote = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .is_ok_and(|mtime| mtime > before);
        if !wrote {
            warn!("agent left no fresh CONTINUITY.md; session stays recoverable");
        }
        wrote
    }

    /// Feed the previous incarnation's `CONTINUITY.md` into a fresh
    /// session as its first turn. Best-effort: a failed or timed-out
    /// inject is a memory-less start, not a failed spawn.
    async fn inject_continuity(&self, client: &AcpClient<BotClientHandler>, session_id: &str) {
        let Ok(text) = std::fs::read_to_string(self.cwd().join(CONTINUITY_FILE)) else {
            return;
        };
        if text.trim().is_empty() {
            return;
        }
        info!("injecting continuity summary into new session");
        if let Err(error) = self
            .maintenance_prompt(client, session_id, format!("{INJECT_PROMPT}\n\n{text}"))
            .await
        {
            warn!(%error, "continuity inject failed");
        }
    }

    /// A prompt with no typing, no nudges, and a hard bound — shutdown
    /// and bootstrap turns must never hang the actor. On timeout the turn
    /// is cancelled so its wire request cannot bleed into later work.
    async fn maintenance_prompt(
        &self,
        client: &AcpClient<BotClientHandler>,
        session_id: &str,
        text: String,
    ) -> Result<(), ClientError> {
        let mut prompt: Pin<
            Box<dyn Future<Output = Result<PromptResult, ClientError>> + Send + '_>,
        > = Box::pin(client.prompt(PromptParams::new(
            session_id,
            vec![ContentBlock::Text(TextContent {
                text,
                annotations: None,
                meta: None,
            })],
        )));
        enum Outcome {
            Done(Result<PromptResult, ClientError>),
            TimedOut,
        }
        match futures_lite::future::or(async { Outcome::Done(prompt.as_mut().await) }, async {
            async_io::Timer::after(HANDOFF_TIMEOUT).await;
            Outcome::TimedOut
        })
        .await
        {
            Outcome::Done(result) => result.map(|_| ()),
            Outcome::TimedOut => {
                cancel_and_settle(client, session_id, &mut prompt).await;
                Err(ClientError::Transport(format!(
                    "maintenance prompt timed out after {HANDOFF_TIMEOUT:?}"
                )))
            }
        }
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
) {
    if let Err(error) = client.cancel(session_id).await {
        debug!(%error, "cancel failed — turn may have ended");
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
/// event in the same envelope the agent already parses, carrying the chat
/// the interrupted turn was answering so the update lands in the right room.
fn nudge_event_text(chat: Option<&ChatKey>) -> String {
    let mut event = serde_json::json!({
        "type": "nudge",
        "ts": crate::sender::epoch_secs(),
        "note": "You have been silent too long — the user is waiting. \
                 Send a short update now, then continue the task you were \
                 doing.",
    });
    if let Some(chat) = chat {
        event["platform"] = serde_json::json!(chat.platform);
        event["chat"] = serde_json::json!(chat.id);
    }
    event.to_string()
}

/// Write the shared `AGENTS.md` and `.devin/mcp_config.json`.
///
/// The MCP config is what connects the agent to the daemon: devin loads
/// project-scope MCP servers from the session's cwd, and `acpbot mcp-bridge`
/// pipes them to the shared MCP endpoint — a unix socket, a loopback TCP
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
(on devin the tools appear as `mcp__chat__*` and `mcp_list_tools` with \
`server_name` = `chat` shows their schemas; on harnesses with a generic \
MCP dispatcher — Antigravity's `call_mcp_tool` — use `ServerName: \"chat\"` \
with the tool name below):

- `send_message` `{text, buttons?, chat?}` — post a new message. Every call \
arrives as one complete, separately-visible message: compose the full \
text first and never use a stream of calls to deliver one thought. \
`buttons` is an array of rows of buttons, each `{text, data}` (a press \
arrives as a `button` event carrying `data`) or `{text, url}` (a link).
- `reply` `{message_id, text, buttons?, chat?}` — quote-reply to a specific \
message; same keyboard shape.
- `send_file` `{path}` or `{file_id, kind}`, plus `caption?`/`chat?` — \
send media: images go as photos (gif as animation), videos as video, \
audio as audio (ogg as a voice note), anything else as a document.
- `send_sticker` `{name}` or `{file_id}`, plus `chat?` — send a native \
sticker. Pack names publish into the bot's own Telegram sticker set on \
first send; a `file_id` resends a sticker an event carried.
- `react` `{message_id, emoji?, is_big?, chat?}` — set your emoji reaction \
on a message (`👍`, `❤`, `🔥`, `😁`, `👎`, `🤔`, …); omit `emoji` to \
remove it, `is_big` plays the large animation. A reaction is often the \
better ack — cheaper than a whole message.
- `edit_message` `{message_id, text, chat?}` — rewrite a message you sent, \
e.g. grow a progress note into the final result.
- `delete_message` `{message_id, chat?}` — delete a message (yours \
anywhere, others' where the bot has delete rights).
- `pin_message` `{message_id, unpin?, notify?, chat?}` — pin to the top of \
the chat (`unpin` removes the pin; `notify: false` pins silently).
- `message_status` `{message_id, chat?}` — whether a message still exists. \
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
- `history` `{since?, until?, limit?, chat?}` — a chat's IM record: every \
message the bot saw arrive and everything it sent there, `ts`/`time`, \
`dir`, `from`, `text`. This is the durable log of what was said — it \
predates and outlives any session of yours. Times take epoch seconds, \
RFC3339, or relative `30m`/`2h`/`7d`; `limit` (default 50) keeps the \
newest. \
- `search_history` `{query, since?, until?, limit?, chat?}` — the IM \
record grepped: case-insensitive match on text, sender name, and command \
fields. When you need to recall what was said — before a restart wiped \
your context or before you ever existed — this is the tool: \"what did we \
say about X yesterday\" is a `search_history` call, not a guess. \
- `chat_info` `{chat}` — look up a chat through your bot identity: \
numeric id, @username, or t.me link (`t.me/name`, `t.me/c/<id>/<msg>`; \
invite links cannot resolve). Returns id, type, title, description, \
member count, and `bot_status` — `readable: true` means the bot sits in \
it, so its events reach you and `history`/`fetch_message` work there. \
Use it when a user links a chat or asks you to look at one — never guess \
at a t.me link's web page; that preview shows you almost nothing. \
- `fetch_message` `{chat, message_id?}` — read one message from a chat \
the bot belongs to: briefly forwards it here to read it, then deletes \
the copy. `chat` is the SOURCE; a message link supplies `message_id`. \
Chats the bot isn't in cannot answer — ask the user to add the bot. \
- `restart` `{}` — reincarnate your process after editing AGENTS.md or \
adding skills. The next spawn is a FRESH session: anything the next you \
must remember goes in AGENTS.md or CONTINUITY.md before you call it.

**One session, every chat.** All of the bot's conversations share this one \
agent: events from every chat arrive here, and `platform` + `chat` on each \
event say where it happened — what a user tells you in DM you also know in \
the group, and vice versa. Every tool marked `chat?` above takes an \
optional `chat` argument: the `chat` id from an event, or `platform:id`. \
Omit it and the tool acts on the chat the current events came from — pass \
it to reach another conversation: carry a group answer into a DM, check a \
group's `history` while answering a private question, or post into a chat \
nobody pinged you in. Cross-chat sends are real sends — do them when the \
user asked for it or the context makes it obviously right, not on a whim.

**Your context resets; the record doesn't.** Every spawn starts a fresh \
session — the daemon restarts you on shutdown, `restart`, or a harness \
swap, and your old context window never comes back. What survives: \
`CONTINUITY.md` in your working directory (the handoff note the previous \
you wrote — the daemon asks for it before every clean close, and you may \
update it any time), everything you keep below the managed block in \
AGENTS.md, and the IM record the history tools read. When a conversation \
references something you don't remember, search the record — never guess, \
never claim it didn't happen.

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

**Delegate work, stay the main thread.** While your turn runs, EVERY chat \
waits — you are this bot's single-threaded UI. So a turn is for talking \
and dispatching, not for grinding: anything beyond a quick answer — \
research, builds, multi-step jobs — goes to a subagent run in the \
background (spawn one with your subagent tool); your turn acks, delegates, \
and ends. When the subagent finishes, deliver its result to the chat that \
asked — your `chat` argument names it.

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
A `reply_to` can carry the same `sticker`/`media` shape — what the quoted \
message held, not just its text. \
The file also downloads into `inbox/` inside your working directory: \
`media.file`/`sticker.file` gives `{path, mime}` — `path` is absolute \
(e.g. `/…/chats/shared/inbox/CAACAgE….jpg`, `image/jpeg`), so open it \
with your file tools exactly as given; do not resolve it yourself — \
harnesses that default relative paths elsewhere would miss the file. \
When your client declared `image`/`audio` prompt support, those payloads \
also reach you as native `image`/`audio` content blocks. Otherwise the \
file is yours: inspect it, extract frames or waveforms with ffmpeg, or \
transcribe it — whatever the event needs.
";

/// Sessions awaiting handoff recovery, `slug → session id`, persisted as
/// JSON in `data_dir/sessions.json`. Empty after a clean shutdown.
fn load_sessions(data_dir: &Path) -> HashMap<String, String> {
    std::fs::read_to_string(data_dir.join("sessions.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_sessions(data_dir: &Path, sessions: &HashMap<String, String>) {
    match serde_json::to_string_pretty(sessions) {
        // Write-then-rename: a crash mid-write must not corrupt the only
        // record the recovery handoff has of the session it should
        // summarize.
        Ok(text) => {
            let tmp = data_dir.join("sessions.json.tmp");
            let target = data_dir.join("sessions.json");
            if let Err(error) =
                std::fs::write(&tmp, &text).and_then(|()| std::fs::rename(&tmp, &target))
            {
                warn!(%error, "failed to persist sessions.json");
            }
        }
        Err(error) => warn!(%error, "failed to serialize sessions"),
    }
}

#[cfg(test)]
pub(super) mod tests {
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
            .join(SHARED_SLUG)
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

    /// A router whose senders record to `out` — every factory call reports
    /// the key it was built for on `keys`.
    pub(crate) fn test_router(
        dir: &Path,
        out: ChanSender<String>,
    ) -> (Arc<ChatRouter>, Receiver<String>) {
        let (keys_tx, keys) = async_channel::unbounded();
        let (spoke_tx, _spoke_rx) = async_channel::bounded(1);
        let shared = Arc::new(AgentShared {
            agent: AgentConfig::default(),
            bridge_bin: PathBuf::from("/nonexistent/acpbot"),
            data_dir: dir.to_path_buf(),
            sticker_dir: dir.join("stickers"),
            persona: None,
            sticker_library: Arc::new(crate::stickerlib::StickerLibrary::load(dir, None).unwrap()),
            sender_for: Box::new(move |key, spoke, _, _| {
                let _ = keys_tx.try_send(key.id.clone());
                Ok(Sender::record(out.clone(), spoke))
            }),
            session_updates: async_channel::unbounded().0,
        });
        (
            Arc::new(ChatRouter::new(
                shared,
                spoke_tx,
                Arc::new(AtomicU64::new(0)),
                "telegram",
            )),
            keys,
        )
    }

    /// `resolve`: bare ids take the daemon's platform, `platform:id` is
    /// explicit, `None` is the turn's chat.
    #[test]
    fn resolve_forms() {
        let dir = std::env::temp_dir().join(format!("acpbot-resolve-{}", std::process::id()));
        let (out, _out_rx) = async_channel::unbounded();
        let (router, _keys) = test_router(&dir, out);

        // No current yet and no argument: nothing to answer.
        assert!(router.resolve(None).is_err());

        let key = router.resolve(Some("123")).unwrap();
        assert_eq!(key.id, "123");
        assert_eq!(key.platform, "telegram");

        let key = router.resolve(Some("telegram:456")).unwrap();
        assert_eq!(key.id, "456");

        // A foreign platform can never be routed.
        assert!(router.resolve(Some("discord:9")).is_err());
        assert!(router.resolve(Some("telegram:")).is_err());

        router.set_current(test_key());
        assert_eq!(router.resolve(None).unwrap(), test_key());
    }

    /// A chat's sender is built once, tagged with its key, and reused —
    /// sends after `set_current` or with an explicit `chat` resolve to the
    /// right conversation.
    #[test]
    fn senders_are_per_chat() {
        let dir = std::env::temp_dir().join(format!("acpbot-senders-{}", std::process::id()));
        let (out, _out_rx) = async_channel::unbounded();
        let (router, keys) = test_router(&dir, out);
        router.set_current(test_key());

        let a = router.sender_for(None).unwrap();
        let b = router.sender_for(Some("7")).unwrap();
        assert_eq!(keys.try_recv().unwrap(), "0");
        assert_eq!(keys.try_recv().unwrap(), "7");
        // Second resolution reuses the cached sender — no rebuild.
        let b2 = router.sender_for(Some("telegram:7")).unwrap();
        assert!(keys.try_recv().is_err());
        drop((a, b, b2));

        // Histories are per chat too.
        router
            .history(&test_key())
            .append_event(&serde_json::json!({"ts":1}));
        router.history(&ChatKey {
            platform: "telegram",
            id: "7".to_string(),
        });
        assert_eq!(router.history(&test_key()).tail(None, None, 10).len(), 1);
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
