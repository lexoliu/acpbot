//! Durable bash-backed watches — the agent's scheduled wake-up.
//!
//! A watch is a bash command the daemon runs on the agent's behalf:
//! every line the command prints to stdout becomes a `watch` event on the
//! ordinary dispatcher channel, and the command's end becomes one final
//! event (`exited`, `failed`, or `cancelled`). Scheduling, polling,
//! webhooks, and file monitoring are all the same primitive — the agent
//! writes the bash, the daemon only guarantees durability:
//!
//! - Definitions live in `data_dir/watchers.json` and are re-run on every
//!   daemon start — a watch survives restarts because the *command* is
//!   replayed, not because process state is restored. Commands that care
//!   about wall-clock time should anchor to absolute times (`sleep
//!   $((TARGET - $(date +%s)))`) so a replayed command resumes correctly.
//! - Events flow through the same channel as platform messages, so they
//!   are journaled, logged to IM history, and re-prompted after a crash
//!   like anything else.
//! - `cancel` kills the watch's whole process group; daemon shutdown
//!   kills them quietly — a shutdown is not a cancellation, so no
//!   `cancelled` events are emitted for it.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;

use async_channel::Sender as ChanSender;
use futures_lite::StreamExt;
use futures_lite::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::chat::{ChatEvent, EventSender, EventWatch};
use crate::error::WatchError;

/// One watch definition — the durable record persisted in `watchers.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchDef {
    /// `w_<hex>` — returned by the `watch` tool, used by `cancel_watch`.
    pub id: String,
    /// The bash command, run via `bash -c`.
    pub command: String,
    /// Why the agent created it — rides on every event the watch emits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Chat the events are routed to (platform-native id).
    pub chat: String,
    /// Forum topic the watch was created in, so its events stay there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<i64>,
    /// Creation time, epoch seconds.
    pub created_ts: i64,
}

/// What `list_watches` reports: the definition plus whether the command
/// is actually running right now.
#[derive(Debug, Serialize)]
pub struct WatchInfo {
    #[serde(flatten)]
    pub def: WatchDef,
    /// The live process id, when the command is up.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
}

/// A live watch's handle: enough to cancel it without touching the task.
struct Active {
    /// The child's process id — also its process-group id, since children
    /// spawn in their own group.
    pid: u32,
    /// Signalled by `cancel`/`shutdown` to interrupt the line loop.
    cancel: ChanSender<()>,
    /// Set by `cancel` so the task can tell "user asked" from "daemon is
    /// going down" (the latter is `shutting_down` on the shared state).
    cancelled: bool,
}

/// Mutable state shared with the watch tasks. Held briefly, never across
/// an `.await` — the events sender is cloned out before use.
struct Inner {
    /// Mirror of `watchers.json` — every mutation rewrites the file.
    defs: Vec<WatchDef>,
    /// Live watches by id.
    active: HashMap<String, Active>,
    /// The dispatcher channel. `None` after `shutdown`: watchers may not
    /// hold a sender alive or the daemon could never leave its run loop.
    events: Option<ChanSender<ChatEvent>>,
    /// Daemon is going down: tasks exit quietly and defs stay on disk so
    /// the next run replays them.
    shutting_down: bool,
}

/// Owns the watch registry and every live watch task.
pub struct Watchers {
    /// `data_dir/watchers.json`.
    path: PathBuf,
    /// The daemon's platform tag — events carry it.
    platform: String,
    inner: Mutex<Inner>,
}

impl Drop for Watchers {
    /// The last handle going away without `shutdown` — a test, or a
    /// daemon that died mid-cleanup — still kills its children: a watch
    /// never outlives the daemon. Killed orphans are reaped by init.
    fn drop(&mut self) {
        for (id, active) in self.inner.get_mut().expect("watchers").active.drain() {
            warn!(id = %id, "watch dropped without shutdown");
            kill_group(active.pid);
        }
    }
}

impl Watchers {
    /// An empty registry; [`Watchers::restore`] replays the file.
    pub fn new(data_dir: &Path, events: ChanSender<ChatEvent>, platform: &str) -> Self {
        Self {
            path: data_dir.join("watchers.json"),
            platform: platform.to_string(),
            inner: Mutex::new(Inner {
                defs: Vec::new(),
                active: HashMap::new(),
                events: Some(events),
                shutting_down: false,
            }),
        }
    }

    /// Re-run every persisted definition. Called once at dispatcher
    /// start; defs whose command can't spawn stay in the file and are
    /// reported on their next event cycle instead of being dropped.
    pub fn restore(self: &std::sync::Arc<Self>) {
        let defs = {
            let mut inner = self.inner.lock().expect("watchers");
            inner.defs = load_defs(&self.path);
            inner.defs.clone()
        };
        for def in defs {
            info!(id = %def.id, command = %def.command, "restoring watch");
            self.spawn_task(def);
        }
    }

    /// Register a new watch: persist it, then spawn. Returns its id.
    pub fn add(
        self: &std::sync::Arc<Self>,
        command: String,
        note: Option<String>,
        chat: String,
        thread_id: Option<i64>,
    ) -> Result<String, WatchError> {
        if command.trim().is_empty() {
            return Err(WatchError::EmptyCommand);
        }
        let mut def = WatchDef {
            id: new_id(),
            command,
            note,
            chat,
            thread_id,
            created_ts: crate::sender::epoch_secs(),
        };
        {
            let mut inner = self.inner.lock().expect("watchers");
            if inner.shutting_down {
                return Err(WatchError::ShuttingDown);
            }
            while inner.defs.iter().any(|d| d.id == def.id) || inner.active.contains_key(&def.id) {
                def.id = new_id();
            }
            inner.defs.push(def.clone());
            save_defs(&self.path, &inner.defs);
        }
        info!(id = %def.id, command = %def.command, "watch added");
        self.spawn_task(def.clone());
        Ok(def.id)
    }

    /// Every definition, annotated with liveness.
    pub fn list(&self) -> Vec<WatchInfo> {
        let inner = self.inner.lock().expect("watchers");
        inner
            .defs
            .iter()
            .map(|def| WatchInfo {
                def: def.clone(),
                pid: inner.active.get(&def.id).map(|a| a.pid),
            })
            .collect()
    }

    /// Stop a watch: mark it cancelled, drop the definition, kill the
    /// process group. The task emits the `cancelled` event itself. A def
    /// with no live process (it never spawned) just disappears — the
    /// cancellation is still a success. Unknown ids are a usage error.
    pub fn cancel(&self, id: &str) -> Result<(), WatchError> {
        let active = {
            let mut inner = self.inner.lock().expect("watchers");
            let had_def = inner.defs.iter().any(|def| def.id == id);
            inner.defs.retain(|def| def.id != id);
            save_defs(&self.path, &inner.defs);
            match inner.active.get_mut(id) {
                Some(active) => {
                    active.cancelled = true;
                    Some((active.pid, active.cancel.clone()))
                }
                None if had_def => None,
                None => return Err(WatchError::Unknown(id.to_string())),
            }
        };
        if let Some((pid, cancel)) = active {
            let _ = cancel.try_send(());
            kill_group(pid);
            info!(%id, %pid, "watch cancelled");
        }
        Ok(())
    }

    /// Daemon shutdown: kill every watch quietly — no `cancelled` events
    /// (the defs stay on disk and the next run replays them) — and drop
    /// the event sender so the dispatcher's channel can close.
    pub fn shutdown(&self) {
        let (pids, cancels) = {
            let mut inner = self.inner.lock().expect("watchers");
            inner.shutting_down = true;
            inner.events = None;
            (
                inner.active.values().map(|a| a.pid).collect::<Vec<_>>(),
                inner
                    .active
                    .values()
                    .map(|a| a.cancel.clone())
                    .collect::<Vec<_>>(),
            )
        };
        for cancel in cancels {
            let _ = cancel.try_send(());
        }
        for pid in &pids {
            kill_group(*pid);
        }
        info!(count = pids.len(), "watches shut down");
    }

    /// Spawn the task that runs one watch to its end.
    fn spawn_task(self: &std::sync::Arc<Self>, def: WatchDef) {
        let watchers = self.clone();
        executor_core::spawn(async move { watchers.run(def).await }).detach();
    }

    /// Run one watch: spawn the command, emit each stdout line, then the
    /// ending event. Cancellation interrupts the read loop and kills the
    /// process group.
    async fn run(&self, def: WatchDef) {
        let mut child = match spawn(&def.command) {
            Ok(child) => child,
            Err(error) => {
                warn!(%error, id = %def.id, "watch failed to spawn");
                // Keep the def — a transient spawn failure shouldn't
                // delete the watch; report it and let the next restart
                // try again.
                self.emit(
                    &def,
                    EventWatch {
                        id: def.id.clone(),
                        note: def.note.clone(),
                        status: "failed".into(),
                        line: None,
                        seq: None,
                        exit_code: None,
                        stderr_tail: Some(format!("spawn failed: {error}")),
                    },
                )
                .await;
                return;
            }
        };
        let pid = child.id();
        let (cancel_tx, cancel_rx) = async_channel::unbounded::<()>();
        {
            let mut inner = self.inner.lock().expect("watchers");
            inner.active.insert(
                def.id.clone(),
                Active {
                    pid,
                    cancel: cancel_tx,
                    cancelled: false,
                },
            );
        }

        // stderr drains concurrently — a chatty command would otherwise
        // block on a full pipe while we sit on stdout.
        let mut stderr_pipe = child.stderr.take();
        let stderr_task = executor_core::spawn(async move {
            let mut buf = Vec::new();
            if let Some(mut pipe) = stderr_pipe.take() {
                let _ = pipe.read_to_end(&mut buf).await;
            }
            buf
        });

        let mut lines = {
            let stdout = child.stdout.take().expect("stdout is piped");
            BufReader::new(stdout).lines()
        };

        enum Out {
            Line(Option<io::Result<String>>),
            Exit(io::Result<std::process::ExitStatus>),
            Cancel,
        }

        let mut status: Option<std::process::ExitStatus> = None;
        let mut stdout_done = false;
        let mut cancelled = false;
        let mut seq = 0u64;
        loop {
            if stdout_done && status.is_some() {
                break;
            }
            // stdout closing early (a command that `exec 1>&-`s or
            // daemonizes) ends only the line stream — the process still
            // runs until it exits or is cancelled.
            let out = if stdout_done {
                futures_lite::future::race(async { Out::Exit(child.status().await) }, async {
                    let _ = cancel_rx.recv().await;
                    Out::Cancel
                })
                .await
            } else if status.is_some() {
                Out::Line(lines.next().await)
            } else {
                futures_lite::future::race(
                    async { Out::Line(lines.next().await) },
                    futures_lite::future::race(async { Out::Exit(child.status().await) }, async {
                        let _ = cancel_rx.recv().await;
                        Out::Cancel
                    }),
                )
                .await
            };
            match out {
                Out::Line(Some(Ok(line))) => {
                    seq += 1;
                    self.emit(
                        &def,
                        EventWatch {
                            id: def.id.clone(),
                            note: def.note.clone(),
                            status: "fired".into(),
                            line: Some(line),
                            seq: Some(seq),
                            exit_code: None,
                            stderr_tail: None,
                        },
                    )
                    .await;
                }
                Out::Line(Some(Err(_))) | Out::Line(None) => stdout_done = true,
                Out::Exit(Ok(exit)) => status = Some(exit),
                Out::Exit(Err(_)) => break,
                Out::Cancel => {
                    cancelled = true;
                    break;
                }
            }
        }

        // Anything that left the loop without an exit status — an
        // explicit cancel, or a stdout that broke while the command
        // lives on — is killed before reaping so `status()` can't hang.
        // `kill` reaches the direct child on every platform; the group
        // kill gets the command's whole tree on unix.
        if status.is_none() {
            kill_group(pid);
            let _ = child.kill();
        }
        let status = match status {
            Some(exit) => Some(exit),
            None => child.status().await.ok(),
        };
        let stderr = String::from_utf8_lossy(&stderr_task.await)
            .trim()
            .to_string();
        let stderr_tail = (!stderr.is_empty()).then(|| tail(&stderr, 2048));

        // Resolve the ending: shutting_down is silent, an explicit cancel
        // says so, anything else is the command's own exit code. The
        // locked block ends before any `.await` — a MutexGuard isn't Send.
        let (shutting_down, was_cancelled) = {
            let mut inner = self.inner.lock().expect("watchers");
            let shutting_down = inner.shutting_down;
            let was_cancelled = inner.active.get(&def.id).is_some_and(|a| a.cancelled) || cancelled;
            inner.active.remove(&def.id);
            if !shutting_down {
                inner.defs.retain(|d| d.id != def.id);
                save_defs(&self.path, &inner.defs);
            }
            (shutting_down, was_cancelled)
        };

        if shutting_down {
            return;
        }
        let (status_name, exit_code) = if was_cancelled {
            ("cancelled", None)
        } else {
            let code = status.and_then(|s| s.code());
            (if code == Some(0) { "exited" } else { "failed" }, code)
        };
        self.emit(
            &def,
            EventWatch {
                id: def.id.clone(),
                note: def.note.clone(),
                status: status_name.into(),
                line: None,
                seq: None,
                exit_code,
                stderr_tail,
            },
        )
        .await;
    }

    /// Push one watch event onto the dispatcher channel. Best-effort:
    /// after shutdown the sender is gone and the event simply doesn't
    /// exist — anything already delivered is journaled.
    async fn emit(&self, def: &WatchDef, watch: EventWatch) {
        let sender = {
            let inner = self.inner.lock().expect("watchers");
            inner.events.clone()
        };
        let Some(sender) = sender else { return };
        let _ = sender
            .send(ChatEvent {
                kind: "watch".into(),
                platform: self.platform.clone(),
                chat: def.chat.clone(),
                ts: crate::sender::epoch_secs(),
                attention: "direct".into(),
                chat_type: None,
                chat_title: None,
                message_id: None,
                from: EventSender {
                    id: "daemon".into(),
                    name: "acpbot".into(),
                },
                text: None,
                command: None,
                button: None,
                reply_to: None,
                sticker: None,
                media: None,
                reaction: None,
                watch: Some(watch),
                link_previews: Vec::new(),
                thread_id: def.thread_id,
            })
            .await;
    }
}

/// `bash -c` with the child in its own process group, so cancelling can
/// kill the command *and* anything it spawned (a `sleep` inside a loop,
/// a pipeline, a forked poller) instead of orphaning it.
fn spawn(command: &str) -> io::Result<async_process::Child> {
    let mut cmd = std::process::Command::new("bash");
    cmd.arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                // Own process group: pgid == child pid → `kill(-pid)`
                // reaches the whole tree.
                libc::setpgid(0, 0);
                Ok(())
            });
        }
    }
    // `Command::from` keeps the std command (pre_exec included) but marks
    // its stdio as unconfigured, so a bare `spawn()` would overwrite the
    // pipes with `inherit` — re-declare them through the async API.
    let mut cmd = async_process::Command::from(cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd.spawn()
}

/// Kill the watch's whole process group on unix, or just the child
/// elsewhere (group semantics don't exist there).
fn kill_group(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

/// `w_` + 12 hex chars — enough entropy that ids are unguessable without
/// pulling in a uuid dependency.
fn new_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let rand = (nanos ^ (std::process::id() as u128) << 64) as u64;
    format!("w_{:012x}", rand & 0xFFFF_FFFF_FFFF)
}

/// The last `max` bytes of `text`, cut on a char boundary.
fn tail(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

/// The durable definitions, or none when the file is absent/corrupt —
/// a corrupt file warns loudly since it means watches silently not
/// running.
fn load_defs(path: &Path) -> Vec<WatchDef> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    match serde_json::from_str(&text) {
        Ok(defs) => defs,
        Err(error) => {
            warn!(%error, "corrupt watchers.json; no watches restored");
            Vec::new()
        }
    }
}

/// Write-then-rename, like `sessions.json`/`inflight.json`: a crash
/// mid-write must not corrupt the watch registry. An empty registry
/// removes the file.
fn save_defs(path: &Path, defs: &[WatchDef]) {
    if defs.is_empty() {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => warn!(%error, "failed to remove watchers.json"),
        }
        return;
    }
    match serde_json::to_string_pretty(defs) {
        Ok(text) => {
            let tmp = path.with_extension("json.tmp");
            if let Err(error) =
                std::fs::write(&tmp, &text).and_then(|()| std::fs::rename(&tmp, path))
            {
                warn!(%error, "failed to persist watchers.json");
            }
        }
        Err(error) => warn!(%error, "failed to serialize watchers"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// `executor_core::spawn` (watch + stderr tasks) needs a global
    /// executor — leak one, unless a sibling test already did.
    fn ensure_executor() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let executor: &'static async_executor::Executor<'static> =
                Box::leak(Box::new(async_executor::Executor::new()));
            if executor_core::try_init_global_executor(executor).is_err() {
                return;
            }
            for _ in 0..2 {
                std::thread::spawn(move || {
                    futures_lite::future::block_on(executor.run(std::future::pending::<()>()));
                });
            }
        });
    }

    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("acpbot-watch-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The next watch event off the channel, or None after `timeout`.
    fn next_event(rx: &async_channel::Receiver<ChatEvent>, timeout: Duration) -> Option<ChatEvent> {
        let started = Instant::now();
        loop {
            match rx.try_recv() {
                Ok(event) => return Some(event),
                Err(_) if started.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => return None,
            }
        }
    }

    fn watchers(dir: &Path) -> (Arc<Watchers>, async_channel::Receiver<ChatEvent>) {
        let (tx, rx) = async_channel::unbounded();
        (Arc::new(Watchers::new(dir, tx, "cli")), rx)
    }

    #[test]
    fn defs_roundtrip_and_empty_removes_file() {
        let dir = tempdir("defs");
        let path = dir.join("watchers.json");
        let def = WatchDef {
            id: "w_1".into(),
            command: "sleep 1".into(),
            note: Some("check it".into()),
            chat: "42".into(),
            thread_id: Some(7),
            created_ts: 1_700_000_000,
        };
        save_defs(&path, std::slice::from_ref(&def));
        let back = load_defs(&path);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].id, "w_1");
        assert_eq!(back[0].thread_id, Some(7));
        assert_eq!(back[0].note.as_deref(), Some("check it"));

        save_defs(&path, &[]);
        assert!(!path.exists(), "empty registry should remove the file");

        std::fs::write(&path, "{not json").unwrap();
        assert!(load_defs(&path).is_empty(), "corrupt file loads empty");
    }

    #[test]
    fn watch_fires_each_line_then_exits() {
        ensure_executor();
        let dir = tempdir("fires");
        let (watchers, rx) = watchers(&dir);
        let id = watchers
            .add(
                "printf 'alpha\\nbravo\\n'".into(),
                Some("two lines".into()),
                "42".into(),
                None,
            )
            .unwrap();

        let first = next_event(&rx, Duration::from_secs(5)).expect("first line");
        assert_eq!(first.kind, "watch");
        assert_eq!(first.chat, "42");
        let watch = first.watch.unwrap();
        assert_eq!(watch.id, id);
        assert_eq!(watch.status, "fired");
        assert_eq!(watch.line.as_deref(), Some("alpha"));
        assert_eq!(watch.seq, Some(1));
        assert_eq!(watch.note.as_deref(), Some("two lines"));

        let second = next_event(&rx, Duration::from_secs(5)).expect("second line");
        assert_eq!(
            second.watch.as_ref().unwrap().line.as_deref(),
            Some("bravo")
        );
        assert_eq!(second.watch.as_ref().unwrap().seq, Some(2));

        let end = next_event(&rx, Duration::from_secs(5)).expect("exit event");
        let watch = end.watch.unwrap();
        assert_eq!(watch.status, "exited");
        assert_eq!(watch.exit_code, Some(0));
        // A finished watch leaves the registry and the file.
        assert!(watchers.list().is_empty());
        assert!(!dir.join("watchers.json").exists());
    }

    #[test]
    fn nonzero_exit_reports_failed_with_stderr() {
        ensure_executor();
        let dir = tempdir("failed");
        let (watchers, rx) = watchers(&dir);
        watchers
            .add("echo oops >&2; exit 3".into(), None, "42".into(), None)
            .unwrap();

        let end = next_event(&rx, Duration::from_secs(5)).expect("exit event");
        let watch = end.watch.unwrap();
        assert_eq!(watch.status, "failed");
        assert_eq!(watch.exit_code, Some(3));
        assert_eq!(watch.stderr_tail.as_deref(), Some("oops"));
    }

    #[test]
    fn cancel_kills_the_command_and_reports() {
        ensure_executor();
        let dir = tempdir("cancel");
        let (watchers, rx) = watchers(&dir);
        let id = watchers
            .add("sleep 60".into(), None, "42".into(), None)
            .unwrap();
        // Let the task register its pid before cancelling.
        std::thread::sleep(Duration::from_millis(300));
        watchers.cancel(&id).unwrap();

        let end = next_event(&rx, Duration::from_secs(5)).expect("cancel event");
        assert_eq!(end.watch.as_ref().unwrap().status, "cancelled");
        assert!(watchers.list().is_empty());
        assert!(!dir.join("watchers.json").exists());
        assert!(watchers.cancel(&id).is_err(), "second cancel is an error");
    }

    #[test]
    fn restore_respawns_persisted_defs() {
        ensure_executor();
        let dir = tempdir("restore");
        save_defs(
            &dir.join("watchers.json"),
            &[WatchDef {
                id: "w_restored".into(),
                command: "echo back".into(),
                note: None,
                chat: "7".into(),
                thread_id: None,
                created_ts: 1,
            }],
        );
        let (watchers, rx) = watchers(&dir);
        watchers.restore();

        let event = next_event(&rx, Duration::from_secs(5)).expect("restored line");
        assert_eq!(event.chat, "7");
        assert_eq!(event.watch.as_ref().unwrap().line.as_deref(), Some("back"));
    }

    #[test]
    fn shutdown_emits_nothing_and_keeps_defs() {
        ensure_executor();
        let dir = tempdir("shutdown");
        let (watchers, rx) = watchers(&dir);
        watchers
            .add("sleep 60".into(), None, "42".into(), None)
            .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        watchers.shutdown();

        // A shutdown kill is not a cancellation: no event, and the def
        // survives so the next run replays it.
        assert!(next_event(&rx, Duration::from_millis(500)).is_none());
        assert_eq!(watchers.list().len(), 1);
        let defs = load_defs(&dir.join("watchers.json"));
        assert_eq!(defs.len(), 1);
    }

    /// A command that closes stdout early (`exec 1>&-`, a daemonizer)
    /// still runs to its real exit — EOF ends the line stream, not the
    /// process.
    #[test]
    fn stdout_eof_early_still_waits_for_exit() {
        ensure_executor();
        let dir = tempdir("eof");
        let (watchers, rx) = watchers(&dir);
        watchers
            .add("exec 1>&-; sleep 0.2".into(), None, "42".into(), None)
            .unwrap();
        let event = next_event(&rx, Duration::from_secs(3)).expect("exit event");
        let watch = event.watch.expect("watch payload");
        assert_eq!(watch.status, "exited");
        assert_eq!(watch.exit_code, Some(0));
    }

    #[test]
    fn empty_command_is_refused() {
        let dir = tempdir("empty");
        let (watchers, _rx) = watchers(&dir);
        assert!(watchers.add("   ".into(), None, "42".into(), None).is_err());
    }

    #[test]
    fn tail_cuts_on_char_boundary() {
        let text = format!("{}中", "a".repeat(3000));
        let cut = tail(&text, 10);
        assert_eq!(cut.len(), 10);
        let short = tail("hola", 10);
        assert_eq!(short, "hola");
    }
}
