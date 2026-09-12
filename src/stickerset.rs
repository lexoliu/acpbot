//! The bot's own Telegram sticker set — where pack files become real
//! stickers.
//!
//! Sending a pack sticker uploads the file into a bot-owned set via
//! `createNewStickerSet`/`addStickerToSet`, then sends it by `file_id` —
//! a native sticker bubble, and the set itself is a normal Telegram pack
//! (`t.me/addstickers/<name>`) users can browse and adopt. `file_id`s are
//! cached in `<pack dir>/published.json` keyed by content hash, so a
//! sticker is uploaded once and a file rewritten under the same name is
//! re-published.
//!
//! Identity resolution is lazy: the first publish calls `getMe` for the
//! bot's `user_id` (the set's owner argument) and username (the mandatory
//! `_by_<bot>` name suffix), so daemon startup never blocks on Telegram.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::error::StickerSetError;
use async_lock::Mutex;
use botkit_core::FileSource;
use botkit_telegram::{NewSticker, TelegramClient};

use crate::stickers::Sticker;

/// Default set-name prefix; the full name is `<prefix>_by_<bot username>`.
const DEFAULT_SET_PREFIX: &str = "acpbot";

/// The bot's sticker set and its publish cache.
///
/// Sticker sets are owned by a *user* — the bot manages the set on their
/// behalf (the `_by_<bot>` name suffix is what grants it edit rights). The
/// owner is `sticker_set_owner` from config when set, else the first
/// private-chat user to trigger a publish (a private chat's id is its
/// user's id).
pub struct StickerSet {
    client: TelegramClient,
    /// The pack directory; `published.json` lives here next to the files.
    dir: PathBuf,
    /// Set name override from config; `None` derives from the bot username.
    name: Option<String>,
    /// Set title override from config.
    title: Option<String>,
    /// Configured owner user id.
    owner: Option<i64>,
    /// Resolved lazily on first publish: the owner `user_id`, the set's
    /// resolved name, and the sticker-name → `file_id` cache.
    state: Mutex<State>,
}

/// Everything that needs the network (or a first write) to initialize.
struct State {
    /// The owning user's Telegram id once resolved.
    user_id: Option<i64>,
    /// The resolved set name once known.
    resolved_name: Option<String>,
    /// `sticker name → publish record`, persisted to `published.json`.
    published: BTreeMap<String, Published>,
}

/// One entry in `published.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Published {
    /// Content hash of the file that was uploaded.
    hash: u64,
    /// The Telegram `file_id` the set gave it.
    file_id: String,
}

impl StickerSet {
    /// A set manager for `client` whose pack files live in `dir`.
    pub fn new(
        client: TelegramClient,
        dir: PathBuf,
        name: Option<String>,
        title: Option<String>,
        owner: Option<i64>,
    ) -> Self {
        Self {
            client,
            dir,
            name,
            title,
            owner,
            state: Mutex::new(State {
                user_id: None,
                resolved_name: None,
                published: BTreeMap::new(),
            }),
        }
    }

    /// Publish `sticker` into the set if needed and return its `file_id`.
    /// `owner_hint` supplies the owning user id when config didn't — the
    /// caller passes the chat id, which equals the user's id in private
    /// chats.
    ///
    /// Serialized on the state lock so concurrent chats cannot interleave
    /// an add with another add's `getStickerSet` tail-lookup.
    pub async fn publish(
        &self,
        sticker: &Sticker,
        owner_hint: Option<i64>,
    ) -> Result<String, StickerSetError> {
        let mut state = self.state.lock().await;
        if state.published.is_empty() {
            state.published = load_published(&self.dir)?;
        }

        let bytes = async_fs::read(&sticker.path).await?;
        let hash = {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            hasher.finish()
        };
        if let Some(entry) = state.published.get(&sticker.name)
            && entry.hash == hash
        {
            return Ok(entry.file_id.clone());
        }

        let (user_id, name) = self.identity(&mut state, owner_hint).await?;
        let filename = sticker
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
            .ok_or_else(|| {
                StickerSetError::Other(format!(
                    "{} has no usable file name",
                    sticker.path.display()
                ))
            })?;
        let new = NewSticker::new(
            FileSource::Bytes(bytes),
            filename,
            sticker.format,
            sticker.emoji.clone(),
        );

        // The set must exist before any edit — the publish shape differs:
        // create (first ever), replace (file changed under a known name),
        // or append (new name in an existing set).
        let set = match self.client.get_sticker_set(&name).await {
            Ok(set) => Some(set),
            Err(error) if is_missing_set(&error) => None,
            Err(error) => return Err(error.into()),
        };
        let old = state.published.get(&sticker.name).cloned();

        // Position in the set the new sticker's `file_id` will occupy after
        // the edit: the replaced sticker's slot, or the end for appends and
        // fresh sets.
        let index = match (&set, &old) {
            (Some(set), Some(old)) => set.stickers.iter().position(|s| s.file_id == old.file_id),
            _ => None,
        };
        match (&set, old) {
            (Some(_), Some(old)) if index.is_some() => {
                self.client
                    .replace_sticker_in_set(user_id, &name, &old.file_id, new)
                    .await?;
            }
            (Some(_), _) => self.client.add_sticker_to_set(user_id, &name, new).await?,
            (None, _) => {
                let title = self
                    .title
                    .clone()
                    .unwrap_or_else(|| format!("{} stickers", DEFAULT_SET_PREFIX));
                self.client
                    .create_sticker_set(user_id, &name, &title, new)
                    .await?;
            }
        }

        let set = self.client.get_sticker_set(&name).await?;
        let file_id = index
            .and_then(|i| set.stickers.get(i))
            .or_else(|| set.stickers.last())
            .map(|s| s.file_id.clone())
            .ok_or_else(|| {
                StickerSetError::Other("sticker set is empty right after publishing".to_string())
            })?;
        state.published.insert(
            sticker.name.clone(),
            Published {
                hash,
                file_id: file_id.clone(),
            },
        );
        save_published(&self.dir, &state.published)?;
        Ok(file_id)
    }

    /// Resolve the set's owner id and name once, then cache both. The
    /// owner comes from config or the caller's hint — never `getMe`,
    /// because sticker sets cannot be owned by the bot itself.
    async fn identity(
        &self,
        state: &mut State,
        owner_hint: Option<i64>,
    ) -> Result<(i64, String), StickerSetError> {
        if state.user_id.is_none() {
            state.user_id = self.owner.or(owner_hint);
            if state.user_id.is_none() {
                return Err(StickerSetError::Other(
                    "sticker sets are owned by a Telegram user — set \
                     `sticker_set_owner` in the config or send the sticker \
                     from a private chat"
                        .to_string(),
                ));
            }
        }
        if state.resolved_name.is_none() {
            let me = self.client.get_me().await?;
            let username = me.username.ok_or_else(|| {
                StickerSetError::Other(
                    "bot account has no username — cannot name a sticker set".to_string(),
                )
            })?;
            state.resolved_name = Some(
                self.name
                    .clone()
                    .unwrap_or_else(|| format!("{DEFAULT_SET_PREFIX}_by_{username}")),
            );
        }
        Ok((
            state.user_id.expect("set above"),
            state.resolved_name.clone().expect("set above"),
        ))
    }
}

/// Telegram's "no such set" error.
fn is_missing_set(error: &botkit_core::BotError) -> bool {
    error.to_string().contains("STICKERSET_INVALID")
}

/// Where the publish cache lives inside the pack directory.
fn published_path(dir: &Path) -> PathBuf {
    dir.join("published.json")
}

fn load_published(dir: &Path) -> Result<BTreeMap<String, Published>, StickerSetError> {
    match std::fs::read_to_string(published_path(dir)) {
        Ok(text) => Ok(serde_json::from_str(&text)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e.into()),
    }
}

fn save_published(
    dir: &Path,
    published: &BTreeMap<String, Published>,
) -> Result<(), StickerSetError> {
    std::fs::write(
        published_path(dir),
        serde_json::to_string_pretty(published)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Publish a real sticker into the bot's real set. Needs
    /// `TELEGRAM_BOT_TOKEN` and `TELEGRAM_STICKER_OWNER` (the user id who
    /// owns the set); creates the `acpbot_by_<bot>` set on first run and
    /// verifies a `file_id` comes back and lands in `published.json`.
    ///
    /// ```sh
    /// TELEGRAM_BOT_TOKEN=... TELEGRAM_STICKER_OWNER=12345 \
    ///     cargo test publish_real_sticker -- --ignored
    /// ```
    #[test]
    #[ignore = "needs TELEGRAM_BOT_TOKEN + TELEGRAM_STICKER_OWNER and network access"]
    fn publish_real_sticker() {
        let token = std::env::var("TELEGRAM_BOT_TOKEN").expect("TELEGRAM_BOT_TOKEN");
        let owner: i64 = std::env::var("TELEGRAM_STICKER_OWNER")
            .expect("TELEGRAM_STICKER_OWNER")
            .parse()
            .expect("TELEGRAM_STICKER_OWNER is a user id");
        let dir = std::env::temp_dir().join(format!("acpbot-livepub-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A spec-conforming static sticker: one side exactly 512px, under
        // the 512KB limit even with stored (uncompressed) deflate blocks.
        let png = test_png(512, 96, [255, 200, 40, 255]);
        std::fs::write(dir.join("test_sun.png"), &png).unwrap();
        std::fs::write(dir.join("stickers.toml"), "test_sun = \"a test sun\"\n").unwrap();

        let pack = crate::stickers::StickerPack::load(&dir).unwrap();
        let sticker = pack.get("test_sun").expect("test_sun in pack");
        assert_eq!(sticker.format, "static");

        let set = StickerSet::new(
            TelegramClient::new(token),
            dir.clone(),
            None,
            None,
            Some(owner),
        );
        let executor = async_executor::Executor::new();
        let file_id = futures_lite::future::block_on(executor.run(set.publish(sticker, None)))
            .expect("publish test_sun");
        assert!(!file_id.is_empty(), "publish returned no file_id");

        // Second publish is a cache hit — same file_id, no upload.
        let again = futures_lite::future::block_on(executor.run(set.publish(sticker, None)))
            .expect("republish");
        assert_eq!(again, file_id);
        let cache = std::fs::read_to_string(dir.join("published.json")).unwrap();
        assert!(cache.contains(&file_id), "published.json: {cache}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A minimal valid WxH solid-color PNG (stored deflate blocks).
    fn test_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        fn crc32(data: &[u8]) -> u32 {
            let mut crc = 0xFFFF_FFFFu32;
            for &b in data {
                crc ^= u32::from(b);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(crc & 1));
                }
            }
            !crc
        }
        fn chunk(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut c = (data.len() as u32).to_be_bytes().to_vec();
            c.extend_from_slice(name);
            c.extend_from_slice(data);
            let crc = crc32(&[&name[..], data].concat());
            c.extend_from_slice(&crc.to_be_bytes());
            c
        }
        fn adler32(data: &[u8]) -> u32 {
            let (mut a, mut b) = (1u32, 0u32);
            for &x in data {
                a = (a + u32::from(x)) % 65521;
                b = (b + a) % 65521;
            }
            (b << 16) | a
        }

        // Raw image data: filter byte 0 per row + RGBA pixels — build one
        // row, then repeat it `h` times.
        let mut row = Vec::with_capacity(w as usize * 4 + 1);
        row.push(0u8);
        for _ in 0..w {
            row.extend_from_slice(&rgba);
        }
        let raw = row.repeat(h as usize);
        // zlib stream: stored (uncompressed) deflate blocks + adler32.
        let mut z = vec![0x78, 0x01];
        let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
        for (i, block) in blocks.iter().enumerate() {
            z.push(u8::from(i + 1 == blocks.len()));
            z.extend_from_slice(&(block.len() as u16).to_le_bytes());
            z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
            z.extend_from_slice(block);
        }
        z.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = w.to_be_bytes().to_vec();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
        png.extend_from_slice(&chunk(b"IHDR", &ihdr));
        png.extend_from_slice(&chunk(b"IDAT", &z));
        png.extend_from_slice(&chunk(b"IEND", &[]));
        png
    }

    #[test]
    fn published_cache_roundtrips() {
        let dir = std::env::temp_dir().join(format!("acpbot-pub-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Missing cache file loads as empty.
        assert!(load_published(&dir).unwrap().is_empty());

        let mut published = BTreeMap::new();
        published.insert(
            "hi".to_string(),
            Published {
                hash: 42,
                file_id: "CAACAgEAAxk".to_string(),
            },
        );
        save_published(&dir, &published).unwrap();

        let loaded = load_published(&dir).unwrap();
        assert_eq!(loaded["hi"].hash, 42);
        assert_eq!(loaded["hi"].file_id, "CAACAgEAAxk");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
