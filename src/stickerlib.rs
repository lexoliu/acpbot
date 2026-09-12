//! The sticker library: a persistent catalog of stickers from Telegram
//! sets the bot does not own.
//!
//! A bot may send *any* sticker Telegram hosts by `file_id` — ownership is
//! only needed to publish into a set. The library exploits that: importing
//! a set (`getStickerSet` returns every sticker's `file_id`, emoji, and
//! format) or saving a sticker an event carried puts its `file_id` in the
//! catalog, and `send_sticker` resolves the catalog name thereafter.
//!
//! State persists in `<data_dir>/sticker_library.json` — bot-global and
//! shared across chats, so a set imported in one conversation is sendable
//! in all of them, and across daemon restarts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_lock::Mutex;
use botkit_telegram::TelegramClient;
use serde::{Deserialize, Serialize};

use crate::error::StickerLibError;

/// One sticker the bot can send by `file_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibrarySticker {
    /// Identifier passed to `sendSticker`.
    pub file_id: String,
    /// The emoji Telegram associates with the sticker, if any.
    pub emoji: Option<String>,
    /// The set it came from, when known.
    pub set_name: Option<String>,
    /// The set's human title, when known.
    pub set_title: Option<String>,
    /// `static`, `animated`, or `video`.
    pub format: String,
}

/// The persisted catalog.
#[derive(Debug, Default, Serialize, Deserialize)]
struct State {
    /// Imported sets: short name → title.
    #[serde(default)]
    sets: BTreeMap<String, String>,
    /// Sendable stickers by catalog name (`<set>:<emoji>`, or a name the
    /// agent chose via `save_sticker`).
    #[serde(default)]
    stickers: BTreeMap<String, LibrarySticker>,
}

/// The shared sticker catalog. Construct once per daemon and clone the
/// `Arc` into every chat's tool set.
pub struct StickerLibrary {
    path: PathBuf,
    /// `None` on the CLI platform — imports need the Telegram API, while
    /// lookup and `save_sticker` stay available.
    client: Option<TelegramClient>,
    state: Mutex<State>,
}

impl StickerLibrary {
    /// Load `<data_dir>/sticker_library.json`; a missing file is an empty
    /// catalog.
    ///
    /// # Errors
    /// [`StickerLibError::Io`]/[`StickerLibError::Parse`] on a corrupt file.
    pub fn load(data_dir: &Path, client: Option<TelegramClient>) -> Result<Self, StickerLibError> {
        let path = data_dir.join("sticker_library.json");
        let state = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| StickerLibError::Parse {
                path: path.clone(),
                source,
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => State::default(),
            Err(source) => {
                return Err(StickerLibError::Io {
                    action: "read",
                    path: path.clone(),
                    source,
                });
            }
        };
        Ok(Self {
            path,
            client,
            state: Mutex::new(state),
        })
    }

    /// Import every sticker in a Telegram set into the catalog.
    ///
    /// `query` accepts the set's short name or a `t.me/addstickers/<name>`
    /// link — both appear in the wild. Stickers are named `<set>:<emoji>`
    /// with a `~n` suffix on collisions; the returned list shows the names
    /// `send_sticker` accepts.
    ///
    /// # Errors
    /// [`StickerLibError::NoClient`] on the CLI platform,
    /// [`StickerLibError::Telegram`] when the set cannot be fetched,
    /// [`StickerLibError::Io`] when the catalog cannot be persisted.
    pub async fn import_set(&self, query: &str) -> Result<(String, Vec<String>), StickerLibError> {
        let name = parse_set_query(query).ok_or_else(|| {
            StickerLibError::BadQuery(format!(
                "{query:?} is not a sticker-set name or a t.me/addstickers link"
            ))
        })?;
        let client = self.client.as_ref().ok_or(StickerLibError::NoClient)?;
        let set = client.get_sticker_set(&name).await?;

        let mut state = self.state.lock().await;
        let mut names = Vec::with_capacity(set.stickers.len());
        for sticker in &set.stickers {
            let format = if sticker.is_animated {
                "animated"
            } else if sticker.is_video {
                "video"
            } else {
                "static"
            };
            let entry = LibrarySticker {
                file_id: sticker.file_id.clone(),
                emoji: sticker.emoji.clone(),
                set_name: Some(set.name.clone()),
                set_title: Some(set.title.clone()),
                format: format.to_string(),
            };
            let base = match &sticker.emoji {
                Some(emoji) => format!("{}:{emoji}", set.name),
                None => format!("{}:#{}", set.name, names.len() + 1),
            };
            let mut name = base.clone();
            let mut n = 2;
            while state.stickers.contains_key(&name) {
                name = format!("{base}~{n}");
                n += 1;
            }
            state.stickers.insert(name.clone(), entry);
            names.push(name);
        }
        state.sets.insert(set.name.clone(), set.title.clone());
        self.save_locked(&state)?;
        Ok((set.title, names))
    }

    /// Save a single sticker under `name` — e.g. one an event carried that
    /// the agent wants to keep without importing its whole set.
    ///
    /// # Errors
    /// [`StickerLibError::NameTaken`] when `name` is already in use,
    /// [`StickerLibError::Io`] when the catalog cannot be persisted.
    pub async fn save_sticker(
        &self,
        name: &str,
        sticker: LibrarySticker,
    ) -> Result<(), StickerLibError> {
        let mut state = self.state.lock().await;
        if state.stickers.contains_key(name) {
            return Err(StickerLibError::NameTaken(name.to_string()));
        }
        state.stickers.insert(name.to_string(), sticker);
        self.save_locked(&state)
    }

    /// The `file_id` a catalog name resolves to.
    pub async fn file_id(&self, name: &str) -> Option<String> {
        self.state
            .lock()
            .await
            .stickers
            .get(name)
            .map(|s| s.file_id.clone())
    }

    /// Catalog entries for `list_stickers`: `(name, format, emoji, set_title)`.
    pub async fn list(&self) -> Vec<(String, String, String, Option<String>)> {
        self.state
            .lock()
            .await
            .stickers
            .iter()
            .map(|(name, s)| {
                (
                    name.clone(),
                    s.format.clone(),
                    s.emoji.clone().unwrap_or_default(),
                    s.set_title.clone(),
                )
            })
            .collect()
    }

    /// Imported sets for `list_sticker_sets`: `(name, title, sticker count)`.
    pub async fn list_sets(&self) -> Vec<(String, String, usize)> {
        let state = self.state.lock().await;
        state
            .sets
            .iter()
            .map(|(name, title)| {
                let count = state
                    .stickers
                    .values()
                    .filter(|s| s.set_name.as_deref() == Some(name))
                    .count();
                (name.clone(), title.clone(), count)
            })
            .collect()
    }

    /// Persist the catalog atomically (tmp file + rename).
    fn save_locked(&self, state: &State) -> Result<(), StickerLibError> {
        let tmp = self.path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(state)?;
        std::fs::write(&tmp, text).map_err(|source| StickerLibError::Io {
            action: "write",
            path: tmp.clone(),
            source,
        })?;
        std::fs::rename(&tmp, &self.path).map_err(|source| StickerLibError::Io {
            action: "rename",
            path: self.path.clone(),
            source,
        })
    }
}

/// Normalize a set reference: `name`, `t.me/addstickers/name`, or
/// `https://t.me/addstickers/name`.
fn parse_set_query(query: &str) -> Option<String> {
    let query = query.trim();
    let name = query
        .strip_prefix("https://t.me/addstickers/")
        .or_else(|| query.strip_prefix("http://t.me/addstickers/"))
        .or_else(|| query.strip_prefix("t.me/addstickers/"))
        .unwrap_or(query);
    let name = name.trim_end_matches('/');
    if name.is_empty() || name.contains('/') || name.contains(char::is_whitespace) {
        return None;
    }
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_set_queries() {
        assert_eq!(parse_set_query("PANDECHENG").as_deref(), Some("PANDECHENG"));
        assert_eq!(
            parse_set_query("https://t.me/addstickers/PANDECHENG").as_deref(),
            Some("PANDECHENG")
        );
        assert_eq!(
            parse_set_query("t.me/addstickers/foo_bar").as_deref(),
            Some("foo_bar")
        );
        assert!(parse_set_query("").is_none());
        assert!(parse_set_query("https://example.com/x").is_none());
        assert!(parse_set_query("a b").is_none());
    }
}
