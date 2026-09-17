//! The per-chat transcript log — `history.jsonl` inside the chat's own
//! directory under `data_dir/chats/` (separate from the shared agent's
//! working directory).
//!
//! Every inbound [`ChatEvent`] (daemon-internal `daemon`/`nudge` notes
//! excluded) and every outbound action the agent takes is appended as one
//! JSON object per line, each carrying `ts` (epoch seconds) and `dir`
//! (`"in"`/`"out"`). The `history` and `search_history` chat tools read it
//! back so the agent can pull a time range or grep the past — the ACP
//! session only remembers what fits its context, this file remembers all
//! of it, and it is *the* record the user can read too.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;
use tracing::warn;

/// Append-only JSONL transcript of one chat, shared by the actor (inbound
/// events), its [`Sender`](crate::sender::Sender) (outbound actions), and
/// the history/search chat tools.
#[derive(Clone)]
pub struct History {
    path: PathBuf,
    /// Message ids a probe found deleted — once marked, their records show
    /// `"deleted": true` forever and they are never probed again. Held
    /// only for set/contains, never across an await.
    deleted: Arc<Mutex<HashSet<i64>>>,
}

impl History {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        // The chat dir is created lazily by `prepare_chat_dir`, but the
        // first inbound event lands before that — the log must exist first.
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            warn!(%error, path = %parent.display(), "failed to create history dir");
        }
        Self {
            path,
            deleted: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Record that `message_id` no longer exists (a probe found it
    /// deleted).
    pub fn mark_deleted(&self, message_id: i64) {
        self.deleted.lock().expect("deleted set").insert(message_id);
    }

    /// Whether `message_id` is known deleted.
    pub fn is_deleted(&self, message_id: i64) -> bool {
        self.deleted
            .lock()
            .expect("deleted set")
            .contains(&message_id)
    }

    /// Append one record. Failures only warn — a transcript hiccup must
    /// never break a send.
    pub fn append(&self, record: &impl Serialize) {
        let line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(error) => {
                warn!(%error, "history record not serializable");
                return;
            }
        };
        let mut file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            Ok(file) => file,
            Err(error) => {
                warn!(%error, path = %self.path.display(), "failed to open history");
                return;
            }
        };
        if let Err(error) = writeln!(file, "{line}") {
            warn!(%error, path = %self.path.display(), "failed to append history");
        }
    }

    /// Inbound event → transcript line. Daemon-internal `daemon`/`nudge`
    /// events are not chat history and are skipped by the caller.
    pub fn append_event(&self, event: &impl Serialize) {
        #[derive(Serialize)]
        struct In<'a, T: Serialize> {
            dir: &'a str,
            #[serde(flatten)]
            event: &'a T,
        }
        self.append(&In { dir: "in", event });
    }

    /// Outbound action → transcript line.
    pub fn append_outbound(&self, record: Value) {
        let mut record = record;
        record["dir"] = Value::from("out");
        record["ts"] = Value::from(now_secs());
        self.append(&record);
    }

    /// Newest `limit` records inside `[since, until)` (either bound may be
    /// `None`), chronological order.
    pub fn tail(&self, since: Option<i64>, until: Option<i64>, limit: usize) -> Vec<Value> {
        let mut hits: Vec<Value> = self
            .read_all()
            .into_iter()
            .filter(|r| in_range(r, since, until))
            .collect();
        if hits.len() > limit {
            hits.drain(..hits.len() - limit);
        }
        hits
    }

    /// Like [`Self::tail`], but only records whose `text`, `from.name` or
    /// `command` fields contain `query` (case-insensitive).
    pub fn search(
        &self,
        query: &str,
        since: Option<i64>,
        until: Option<i64>,
        limit: usize,
    ) -> Vec<Value> {
        let needle = query.to_lowercase();
        let matches = |r: &Value| {
            [
                "text",
                "caption",
                "command.name",
                "command.args",
                "from.name",
            ]
            .iter()
            .filter_map(|key| r.pointer(&dot_pointer(key)).and_then(Value::as_str))
            .any(|s| s.to_lowercase().contains(&needle))
        };
        let mut hits: Vec<Value> = self
            .read_all()
            .into_iter()
            .filter(|r| in_range(r, since, until) && matches(r))
            .collect();
        if hits.len() > limit {
            hits.drain(..hits.len() - limit);
        }
        hits
    }

    /// Every parsed line in file order; corrupt lines are skipped.
    fn read_all(&self) -> Vec<Value> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => text
                .lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

fn in_range(r: &Value, since: Option<i64>, until: Option<i64>) -> bool {
    let ts = r["ts"].as_i64().unwrap_or(0);
    since.is_none_or(|s| ts >= s) && until.is_none_or(|u| ts < u)
}

fn dot_pointer(key: &str) -> String {
    format!("/{}", key.replace('.', "/"))
}

/// Current epoch seconds.
fn now_secs() -> i64 {
    crate::sender::epoch_secs()
}

/// A `since`/`until` argument from the agent: epoch seconds, RFC3339
/// (`2026-09-12T10:00:00Z`), or a relative duration back from now
/// (`30m`, `2h`, `7d`).
pub fn parse_time_arg(arg: &str, now: i64) -> Result<i64, String> {
    let arg = arg.trim();
    if let Ok(epoch) = arg.parse::<i64>() {
        return Ok(epoch);
    }
    if let Ok(ts) = arg.parse::<jiff::Timestamp>() {
        return Ok(ts.as_second());
    }
    let (digits, unit) = arg.split_at(arg.len().saturating_sub(1));
    let n: i64 = digits.parse().map_err(|_| {
        format!("unrecognized time {arg:?} — use epoch seconds, RFC3339, or 30m/2h/7d")
    })?;
    let secs = match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86400,
        _ => return Err(format!("unrecognized time unit in {arg:?} — use s/m/h/d")),
    };
    Ok(now - secs)
}

/// The transcript file for a chat working directory.
pub fn history_path(chat_dir: &Path) -> PathBuf {
    chat_dir.join("history.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_tail_and_search() {
        let dir = std::env::temp_dir().join(format!("acpbot-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.jsonl");
        let _ = std::fs::remove_file(&path);
        let history = History::open(&path);

        history.append(&json!({"ts": 100, "dir": "in", "type": "message",
            "text": "morning all", "from": {"id": "u1", "name": "Ada"}}));
        history.append(&json!({"ts": 200, "dir": "out", "type": "message",
            "text": "hey there"}));
        history.append(&json!({"ts": 300, "dir": "in", "type": "message",
            "text": "lunch at noon?", "from": {"id": "u1", "name": "Ada"}}));

        assert_eq!(history.tail(None, None, 10).len(), 3);
        assert_eq!(history.tail(Some(150), None, 10).len(), 2);
        assert_eq!(history.tail(None, Some(200), 10).len(), 1);
        assert_eq!(history.tail(Some(100), Some(300), 10).len(), 2);
        assert_eq!(history.tail(None, None, 2).len(), 2); // newest kept
        assert_eq!(history.tail(None, None, 2)[0]["ts"], json!(200));

        assert_eq!(history.search("lunch", None, None, 10).len(), 1);
        assert_eq!(history.search("ADA", None, None, 10).len(), 2); // name match
        assert_eq!(history.search("hey", None, None, 10).len(), 1);
        assert_eq!(history.search("zzz", None, None, 10).len(), 0);
        assert_eq!(history.search("lunch", None, Some(250), 10).len(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_time_arg_forms() {
        let now = 1_000_000;
        assert_eq!(parse_time_arg("500000", now).unwrap(), 500000);
        assert_eq!(parse_time_arg("30m", now).unwrap(), now - 1800);
        assert_eq!(parse_time_arg("2h", now).unwrap(), now - 7200);
        assert_eq!(parse_time_arg("7d", now).unwrap(), now - 604800);
        let expected = "2026-09-12T10:00:00Z"
            .parse::<jiff::Timestamp>()
            .unwrap()
            .as_second();
        assert_eq!(
            parse_time_arg("2026-09-12T10:00:00Z", now).unwrap(),
            expected
        );
        assert!(parse_time_arg("soon", now).is_err());
    }
}
