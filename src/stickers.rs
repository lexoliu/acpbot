//! The sticker pack: a directory of sticker files the agent sends by name.
//!
//! Layout: every sticker file in the directory is a sticker; its file
//! stem is its name (`thumbs_up.png` → `thumbs_up`). `.png`/`.webp`
//! are static stickers, `.tgs` animated, `.webm` video. Sending publishes
//! the file into the bot's own Telegram sticker set (`stickerset.rs`) and
//! goes out by `file_id` — a real native sticker, not a photo attachment.
//!
//! A `stickers.toml` in the directory annotates entries, either as a
//! one-line meaning or a table with meaning and emoji:
//!
//! ```toml
//! thumbs_up = "approval, agreement"
//! facepalm = { meaning = "exasperation", emoji = "🤦" }
//! ```
//!
//! `published.json` (written by [`crate::stickerset`]) caches the Telegram
//! `file_id` per file content so unchanged stickers are not re-uploaded.

use crate::error::StickersError;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One sticker in the pack.
#[derive(Debug)]
pub struct Sticker {
    /// The name tools refer to it by (file stem).
    pub name: String,
    /// Image file on disk.
    pub path: PathBuf,
    /// Telegram sticker format: `static`, `animated`, or `video`.
    pub format: &'static str,
    /// Optional human/agent-facing description from `stickers.toml`.
    pub meaning: Option<String>,
    /// The emoji the sticker is associated with inside the set.
    pub emoji: String,
}

/// The loaded pack, looked up by name.
#[derive(Debug)]
pub struct StickerPack {
    stickers: BTreeMap<String, Sticker>,
}

/// The Telegram sticker format for a file extension, or `None` for files
/// that cannot be stickers at all. Telegram accepts only `.webp`/`.png` for
/// static stickers — other images send through `send_file` as photos.
fn sticker_format(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|e| e.to_str())? {
        "tgs" => Some("animated"),
        "webm" => Some("video"),
        "webp" | "png" => Some("static"),
        _ => None,
    }
}

impl StickerPack {
    /// Scan `dir` and load the pack. A missing directory is an empty pack.
    /// Scan `dir` and load the pack. A missing directory is an empty pack.
    ///
    /// # Errors
    /// [`StickersError::Io`] on filesystem failures, [`StickersError::Parse`]
    /// when `stickers.toml` is invalid.
    pub fn load(dir: &Path) -> Result<Self, StickersError> {
        let annotations = load_annotations(dir)?;

        let mut stickers = BTreeMap::new();
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(format) = sticker_format(&path) else {
                    continue;
                };
                let Some(stem) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                else {
                    continue;
                };
                let annotation = annotations.get(&stem);
                stickers.insert(
                    stem.clone(),
                    Sticker {
                        name: stem,
                        path,
                        format,
                        meaning: annotation.and_then(|a| a.meaning.clone()),
                        emoji: annotation
                            .and_then(|a| a.emoji.clone())
                            .unwrap_or_else(|| "😀".to_string()),
                    },
                );
            }
        }
        Ok(Self { stickers })
    }

    /// Look up a sticker by name.
    pub fn get(&self, name: &str) -> Option<&Sticker> {
        self.stickers.get(name)
    }

    /// How many stickers the pack holds.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.stickers.len()
    }

    /// JSON array describing the pack, for `list_stickers` output.
    pub fn list_json(&self) -> String {
        #[derive(serde::Serialize)]
        struct Entry<'a> {
            name: &'a str,
            format: &'a str,
            emoji: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            meaning: Option<&'a str>,
        }
        let entries: Vec<Entry<'_>> = self
            .stickers
            .values()
            .map(|s| Entry {
                name: &s.name,
                format: s.format,
                emoji: &s.emoji,
                meaning: s.meaning.as_deref(),
            })
            .collect();
        serde_json::to_string(&entries).expect("sticker list serializes")
    }
}

/// What a `stickers.toml` entry can carry.
#[derive(Default)]
struct Annotation {
    meaning: Option<String>,
    emoji: Option<String>,
}

/// `name → {meaning, emoji}` annotations from `stickers.toml`.
fn load_annotations(dir: &Path) -> Result<BTreeMap<String, Annotation>, StickersError> {
    /// Entries are either a bare meaning string or a table.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Entry {
        Meaning(String),
        Table {
            meaning: Option<String>,
            emoji: Option<String>,
        },
    }
    #[derive(Deserialize)]
    struct Annotations {
        #[serde(flatten)]
        entries: BTreeMap<String, Entry>,
    }

    let path = dir.join("stickers.toml");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(e.into()),
    };
    let entries = toml::from_str::<Annotations>(&text)
        .map_err(|source| StickersError::Parse {
            path: path.clone(),
            source,
        })?
        .entries;
    Ok(entries
        .into_iter()
        .map(|(name, entry)| {
            let annotation = match entry {
                Entry::Meaning(meaning) => Annotation {
                    meaning: Some(meaning),
                    ..Annotation::default()
                },
                Entry::Table { meaning, emoji } => Annotation { meaning, emoji },
            };
            (name, annotation)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_pack_and_meanings() {
        let dir = std::env::temp_dir().join(format!("acpbot-stickers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hi.webp"), b"x").unwrap();
        std::fs::write(dir.join("meme.png"), b"x").unwrap();
        std::fs::write(dir.join("dance.tgs"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"not a sticker").unwrap();
        std::fs::write(dir.join("published.json"), "{}").unwrap();
        std::fs::write(
            dir.join("stickers.toml"),
            "hi = \"greeting\"\nmeme = { meaning = \"funny\", emoji = \"😂\" }\n",
        )
        .unwrap();

        let pack = StickerPack::load(&dir).unwrap();
        assert_eq!(pack.len(), 3);
        let hi = pack.get("hi").unwrap();
        assert_eq!(hi.format, "static");
        assert_eq!(hi.meaning.as_deref(), Some("greeting"));
        assert_eq!(hi.emoji, "😀");
        let meme = pack.get("meme").unwrap();
        assert_eq!(meme.emoji, "😂");
        assert_eq!(pack.get("dance").unwrap().format, "animated");
        assert!(pack.get("notes").is_none());
        assert!(pack.get("published").is_none());
        assert!(pack.get("missing").is_none());
        assert!(pack.list_json().contains("greeting"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_dir_is_empty_pack() {
        let pack = StickerPack::load(Path::new("/nonexistent/acpbot-test")).unwrap();
        assert_eq!(pack.len(), 0);
    }
}
