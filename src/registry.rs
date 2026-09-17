//! The durable registry of every chat the bot holds state for —
//! `chats.json` at the data-dir root.
//!
//! No chat platform offers the bot an enumeration API, so the registry is
//! built by observation: every inbound event notes its chat (with whatever
//! metadata the event carried — type, title, forum topic), and every
//! outbound action to a chat the daemon has never seen registers it too.
//! The [`crate::tools`] `list_chats` tool reads it back so the agent can
//! enumerate its own conversations — "which chats am I in" is a
//! `list_chats` call, not a guess.
//!
//! Keys are the canonical `platform:id` string every tool's `chat`
//! argument accepts; records are what an event can tell us passively —
//! live metadata (member counts, descriptions) is `chat_info`'s job.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::chat::{ChatEvent, ChatKey};

/// What the daemon has observed about one chat. All fields except the
/// timestamps are best-effort — a chat that only ever received outbound
/// sends has neither type nor title until an event arrives or the agent
/// fills the gap with `chat_info`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRecord {
    /// `private`/`group`/`supergroup`/`channel`, when an event reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_type: Option<String>,
    /// The chat's title, when an event reported it (Telegram groups and
    /// channels carry it; DMs and the other platforms do not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// First observed activity, epoch seconds.
    pub first_seen: i64,
    /// Latest observed activity, epoch seconds.
    pub last_seen: i64,
    /// Forum topics the chat's events arrived in: `thread_id → last_seen`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub topics: BTreeMap<i64, i64>,
}

/// The registry itself: a `platform:id → ChatRecord` map persisted
/// atomically (temp file + rename, like `sessions.json`) on every change.
pub struct ChatRegistry {
    path: PathBuf,
    records: BTreeMap<String, ChatRecord>,
    dirty: bool,
}

impl ChatRegistry {
    /// Load `data_dir/chats.json`, backfilling chats that predate the
    /// registry: any `chats/<slug>/` directory without a record is one the
    /// bot already touched — its IM record exists, so the registry should
    /// say so. The slug is `platform`-`-sanitized id`; the ids platforms
    /// actually issue (digits, `-`, `_`, `.`) survive the round trip.
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join("chats.json");
        let mut registry = Self {
            records: std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or_default(),
            path,
            dirty: false,
        };
        if let Ok(dirs) = std::fs::read_dir(data_dir.join("chats")) {
            for entry in dirs.flatten() {
                let Some(slug) = entry.file_name().to_str().map(str::to_owned) else {
                    continue;
                };
                if slug == crate::agent::SHARED_SLUG {
                    continue;
                }
                let Some((platform, id)) = slug.split_once('-') else {
                    continue;
                };
                let key = format!("{platform}:{id}");
                if registry.records.contains_key(&key) {
                    continue;
                }
                let ts = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs() as i64);
                registry.records.insert(
                    key,
                    ChatRecord {
                        chat_type: None,
                        title: None,
                        first_seen: ts,
                        last_seen: ts,
                        topics: BTreeMap::new(),
                    },
                );
                registry.dirty = true;
            }
        }
        registry.flush();
        registry
    }

    /// Note an inbound event's chat: creates the record on first sight and
    /// refreshes `last_seen`, `chat_type`, `title`, and the forum topic
    /// the event arrived in.
    pub fn note_event(&mut self, event: &ChatEvent) {
        let key = ChatKey {
            platform: event.platform.clone(),
            id: event.chat.clone(),
        };
        self.note(
            &key,
            event.ts,
            event.chat_type.as_deref(),
            event.chat_title.as_deref(),
            event.thread_id,
        );
    }

    /// Register a chat that only outbound activity touched (a send or a
    /// probe to an id no event ever came from). `chat_type`/`title` stay
    /// unknown until an event or `chat_info` fills them.
    pub fn note_outbound(&mut self, key: &ChatKey, now: i64) {
        self.note(key, now, None, None, None);
    }

    fn note(
        &mut self,
        key: &ChatKey,
        ts: i64,
        chat_type: Option<&str>,
        title: Option<&str>,
        thread_id: Option<i64>,
    ) {
        let record = self
            .records
            .entry(key.to_string())
            .or_insert_with(|| ChatRecord {
                chat_type: None,
                title: None,
                first_seen: ts,
                last_seen: ts,
                topics: BTreeMap::new(),
            });
        record.first_seen = record.first_seen.min(ts);
        record.last_seen = record.last_seen.max(ts);
        if let Some(chat_type) = chat_type {
            record.chat_type = Some(chat_type.to_string());
        }
        if let Some(title) = title {
            record.title = Some(title.to_string());
        }
        if let Some(thread_id) = thread_id {
            record.topics.insert(thread_id, ts);
        }
        self.dirty = true;
    }

    /// Persist when anything changed — atomic so a crash mid-write keeps
    /// the previous registry rather than a truncated one.
    pub fn flush(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let Ok(text) = serde_json::to_string_pretty(&self.records) else {
            return;
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, &text).is_err() || std::fs::rename(&tmp, &self.path).is_err() {
            warn!(path = %self.path.display(), "failed to persist chat registry");
        }
    }

    /// Every known chat, keyed `platform:id`, for `list_chats`.
    pub fn records(&self) -> &BTreeMap<String, ChatRecord> {
        &self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::EventSender;

    /// A throwaway data dir per test.
    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("acpbot-registry-test-{label}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test dir");
        dir
    }

    fn event(chat: &str, ts: i64) -> ChatEvent {
        ChatEvent {
            kind: "message".into(),
            platform: "telegram".into(),
            chat: chat.to_string(),
            ts,
            attention: "direct".into(),
            chat_type: None,
            chat_title: None,
            message_id: None,
            from: EventSender {
                id: "1".to_string(),
                name: "Tester".to_string(),
            },
            text: None,
            command: None,
            button: None,
            reply_to: None,
            sticker: None,
            media: None,
            reaction: None,
            thread_id: None,
        }
    }

    #[test]
    fn note_event_registers_and_updates() {
        let dir = tempdir("note");
        let mut registry = ChatRegistry::load(&dir);
        assert!(registry.records().is_empty());

        let mut first = event("-100", 100);
        first.chat_type = Some("supergroup".into());
        first.chat_title = Some("The Group".to_string());
        first.thread_id = Some(7);
        registry.note_event(&first);

        // A second event refreshes `last_seen`, keeps `first_seen`, and a
        // newer title replaces the old one.
        let mut second = event("-100", 200);
        second.chat_title = Some("The Renamed Group".to_string());
        second.thread_id = Some(9);
        registry.note_event(&second);

        let record = &registry.records()["telegram:-100"];
        assert_eq!(record.first_seen, 100);
        assert_eq!(record.last_seen, 200);
        assert_eq!(record.chat_type.as_deref(), Some("supergroup"));
        assert_eq!(record.title.as_deref(), Some("The Renamed Group"));
        assert_eq!(
            record.topics.keys().copied().collect::<Vec<_>>(),
            vec![7, 9]
        );

        // An event carrying no metadata must not blank what was learned.
        registry.note_event(&event("-100", 300));
        let record = &registry.records()["telegram:-100"];
        assert_eq!(record.last_seen, 300);
        assert_eq!(record.title.as_deref(), Some("The Renamed Group"));
    }

    #[test]
    fn registry_survives_reload() {
        let dir = tempdir("reload");
        {
            let mut registry = ChatRegistry::load(&dir);
            registry.note_outbound(
                &ChatKey {
                    platform: "telegram".into(),
                    id: "42".to_string(),
                },
                500,
            );
            registry.flush();
        }
        let registry = ChatRegistry::load(&dir);
        let record = &registry.records()["telegram:42"];
        assert_eq!(record.first_seen, 500);
        assert_eq!(record.last_seen, 500);
        assert!(record.title.is_none());
    }

    #[test]
    fn load_backfills_existing_chat_dirs() {
        let dir = tempdir("backfill");
        std::fs::create_dir_all(dir.join("chats/telegram--1002495551562")).expect("chat dir");
        // The shared session dir is agent state, not a chat.
        std::fs::create_dir_all(dir.join("chats/shared")).expect("shared dir");
        let registry = ChatRegistry::load(&dir);
        assert!(registry.records().contains_key("telegram:-1002495551562"));
        assert_eq!(registry.records().len(), 1);
    }
}
