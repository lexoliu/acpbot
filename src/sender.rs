//! Platform-bound message sending for the agent's tools.
//!
//! A [`Sender`] is built per chat, on demand, by the shared
//! [`ChatRouter`](crate::agent::ChatRouter): a tool's `chat` argument (or
//! the in-flight turn's triggering chat) resolves to the sender bound to
//! that conversation.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64};

use async_channel::Sender as ChanSender;
use botkit_core::action::{ChatAction, ChatActionGuard};
use botkit_core::{BotError, FileSource};
use botkit_discord::DiscordClient;
use botkit_discord::action::DiscordActionSender;
use botkit_telegram::action::TelegramActionSender;
use botkit_telegram::{ChatRef, InlineKeyboardMarkup, MediaKind, ReplyMarkup, TelegramClient};
use tracing::warn;

use botkit_cli::wire::{
    Outbound, OutboundAction, OutboundDelete, OutboundEdit, OutboundFile, OutboundMessage,
    OutboundPin, OutboundReaction, WireButton,
};
use botkit_cli::{CliActionSender, CliHub};

use crate::chat::{ChatKey, EventFile};
use crate::error::SenderError;
use crate::stickers::Sticker;
use crate::stickerset::StickerSet;

/// Outbound operations bound to one chat.
///
/// `spoke` is the actor-global signal shared by every `Sender`: the first
/// successful (or attempted) outbound action in a turn posts to it so the
/// turn's typing indicator can stop immediately instead of running until
/// the agent's turn ends.
///
/// `thread` is the forum topic the current event batch arrived in (`0` =
/// none): the actor updates it per turn so sends land in the topic the
/// user is looking at without the agent having to echo an id.
#[derive(Clone)]
pub struct Sender {
    platform: Platform,
    spoke: ChanSender<()>,
    thread: Arc<AtomicI64>,
    /// Epoch milliseconds of the agent's last outbound action — the actor's
    /// nudge watchdog measures user-visible silence against it.
    last_action: Arc<AtomicU64>,
    /// The chat's transcript log; every successful outbound action is
    /// appended so `history`/`search_history` see both directions.
    history: Option<Arc<crate::history::History>>,
}

/// `SystemTime` milliseconds since the epoch, `0` before epoch (never in
/// practice). Wall-clock jumps shift the nudge window slightly; for a
/// 10-second threshold that is noise.
pub(crate) fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// `SystemTime` seconds since the epoch — the `ts` on events and history
/// records.
pub(crate) fn epoch_secs() -> i64 {
    (epoch_ms() / 1000) as i64
}

/// The Bot API refuses `getFile` on anything over 20MB, so larger downloads
/// never start.
const DOWNLOAD_LIMIT: usize = 20 * 1024 * 1024;

/// How long the typing indicator shows between consecutive sends — a beat,
/// not a stall: long enough to read as typing, short enough that a
/// four-message answer costs about two seconds.
const INTER_MESSAGE_BEAT: std::time::Duration = std::time::Duration::from_millis(700);

/// A text blob longer than this is split into smaller bubbles at single
/// newlines, then at sentence-ending punctuation. Sized to what a person
/// actually sends: a line or two.
const BUBBLE_MAX_CHARS: usize = 80;

/// Byte offsets just after every `\n` that sits outside a ``` fenced
/// block — the positions where a split never cuts code in half. A fence
/// toggles on lines whose first non-space characters are ```.
fn fenced_line_ends(text: &str) -> Vec<usize> {
    let mut points = vec![0];
    let mut in_fence = false;
    let mut line_start = 0;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            if text[line_start..i].trim_start().starts_with("```") {
                in_fence = !in_fence;
            }
            if !in_fence {
                points.push(i + 1);
            }
            line_start = i + 1;
        }
    }
    points.push(text.len());
    points
}

/// Split a message into chat bubbles the way a person hits enter: at
/// blank lines first, then — for any still-overlong piece — at single
/// newlines, then at `。！？!?` sentence ends. Fenced code blocks are
/// never cut open. Returns at least one non-empty slice.
fn split_bubbles(text: &str) -> Vec<&str> {
    // Blank lines: segments between fenced line ends that are all
    // whitespace separate paragraphs.
    let points = fenced_line_ends(text);
    let mut paragraphs = Vec::new();
    let mut start = 0;
    for w in points.windows(2) {
        if text[w[0]..w[1]].trim().is_empty() {
            let para = text[start..w[0]].trim();
            if !para.is_empty() {
                paragraphs.push(para);
            }
            start = w[1];
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        paragraphs.push(tail);
    }

    let mut out = Vec::new();
    for para in paragraphs {
        if para.chars().count() <= BUBBLE_MAX_CHARS {
            out.push(para);
            continue;
        }
        // Overlong paragraph: split at single newlines outside fences.
        let lp = fenced_line_ends(para);
        for w in lp.windows(2) {
            let line = para[w[0]..w[1]].trim();
            if line.is_empty() {
                continue;
            }
            // A fenced block stays whole even when it is long.
            if line.chars().count() <= BUBBLE_MAX_CHARS || line.starts_with("```") {
                out.push(line);
            } else {
                out.extend(split_sentences(line));
            }
        }
    }
    if out.is_empty() {
        out.push(text.trim());
    }
    out
}

/// Split at sentence ends — `。！？!?…`, plus `.` when followed by a
/// space or end of text (so `v1.2` and URLs survive) — keeping the
/// punctuation with its sentence.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary = match c {
            '。' | '！' | '？' | '!' | '?' | '…' => true,
            '.' => chars.peek().is_none_or(|(_, n)| n.is_whitespace()),
            _ => false,
        };
        if boundary {
            let piece = text[start..i + c.len_utf8()].trim();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = i + c.len_utf8();
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// Per-platform send implementations.
#[derive(Clone)]
enum Platform {
    /// Telegram.
    Telegram(TelegramSender),
    /// The CLI debug platform: outbound calls become wire-protocol lines.
    Cli(CliSender),
    /// Discord.
    Discord(DiscordSender),
    /// Records outbound calls instead of sending them; tests only.
    #[cfg(test)]
    Record(ChanSender<String>),
}

/// CLI-platform outbound operations bound to one chat.
#[derive(Clone)]
pub struct CliSender {
    hub: CliHub,
    chat: String,
}

/// Discord outbound operations bound to one channel.
#[derive(Clone)]
pub struct DiscordSender {
    client: DiscordClient,
    channel_id: String,
}

/// Telegram outbound operations bound to one chat.
#[derive(Clone)]
pub struct TelegramSender {
    client: TelegramClient,
    chat_id: i64,
    /// The bot's own user id — `getChatMember` probes membership with it.
    bot_id: i64,
    /// The bot-owned sticker set pack files publish into.
    stickers: Arc<StickerSet>,
}

impl Sender {
    /// A sender bound to the Telegram chat the key names.
    pub fn telegram(
        client: TelegramClient,
        key: &ChatKey,
        spoke: ChanSender<()>,
        thread: Arc<AtomicI64>,
        last_action: Arc<AtomicU64>,
        stickers: Arc<StickerSet>,
        bot_id: i64,
    ) -> Result<Self, SenderError> {
        debug_assert_eq!(key.platform, "telegram");
        let chat_id: i64 = key
            .id
            .parse()
            .map_err(|_| SenderError::NonNumericChat(key.id.to_string()))?;
        Ok(Self {
            platform: Platform::Telegram(TelegramSender {
                client,
                chat_id,
                bot_id,
                stickers,
            }),
            spoke,
            thread,
            last_action,
            history: None,
        })
    }

    /// A sender bound to the CLI-platform chat the key names.
    pub fn cli(
        hub: CliHub,
        key: &ChatKey,
        spoke: ChanSender<()>,
        thread: Arc<AtomicI64>,
        last_action: Arc<AtomicU64>,
    ) -> Self {
        debug_assert_eq!(key.platform, "cli");
        Self {
            platform: Platform::Cli(CliSender {
                hub,
                chat: key.id.to_string(),
            }),
            spoke,
            thread,
            last_action,
            history: None,
        }
    }

    /// A sender bound to the Discord channel the key names.
    pub fn discord(
        client: DiscordClient,
        key: &ChatKey,
        spoke: ChanSender<()>,
        thread: Arc<AtomicI64>,
        last_action: Arc<AtomicU64>,
    ) -> Self {
        debug_assert_eq!(key.platform, "discord");
        Self {
            platform: Platform::Discord(DiscordSender {
                client,
                channel_id: key.id.to_string(),
            }),
            spoke,
            thread,
            last_action,
            history: None,
        }
    }

    /// A sender that records every outbound call on `out` instead of
    /// touching a platform API.
    #[cfg(test)]
    pub fn record(out: ChanSender<String>, spoke: ChanSender<()>) -> Self {
        Self {
            platform: Platform::Record(out),
            spoke,
            thread: Arc::new(AtomicI64::new(0)),
            last_action: Arc::new(AtomicU64::new(0)),
            history: None,
        }
    }

    /// Attach the chat's transcript log — every successful outbound
    /// action is appended from then on.
    pub fn with_history(mut self, history: Arc<crate::history::History>) -> Self {
        self.history = Some(history);
        self
    }

    /// Append one outbound action to the transcript (no-op without a
    /// history attached).
    fn log_outbound(&self, record: serde_json::Value) {
        if let Some(history) = &self.history {
            history.append_outbound(record);
        }
    }

    /// Notify the chat that the agent produced output this turn.
    fn spoke(&self) {
        self.last_action
            .store(epoch_ms(), std::sync::atomic::Ordering::Relaxed);
        let _ = self.spoke.try_send(());
    }

    /// Re-arm the typing indicator between consecutive sends.
    ///
    /// The turn's first action already had the typing guard, but each
    /// message that lands clears "typing…", so a second send without a beat
    /// just appears — the pause in between is what makes a multi-message
    /// answer read like a person typing between bubbles. The `spoke`
    /// channel holds one slot: a successful `try_send` means this is the
    /// turn's first action and no beat is needed.
    async fn inter_message_pause(&self) {
        if self.spoke.try_send(()).is_ok() {
            return;
        }
        match &self.platform {
            Platform::Telegram(inner) => {
                let _ = inner
                    .client
                    .send_chat_action(inner.chat_id, "typing", self.thread())
                    .await;
            }
            Platform::Cli(inner) => {
                inner.hub.emit(Outbound::Action(OutboundAction {
                    chat: inner.chat.clone(),
                    action: "typing".to_string(),
                    clear: false,
                    thread_id: self.thread(),
                }));
            }
            Platform::Discord(inner) => {
                let _ = inner.client.trigger_typing(&inner.channel_id).await;
            }
            #[cfg(test)]
            Platform::Record(_) => return,
        }
        async_io::Timer::after(INTER_MESSAGE_BEAT).await;
    }

    /// The forum topic the current event batch arrived in, if any.
    fn thread(&self) -> Option<i64> {
        match self.thread.load(std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            id => Some(id),
        }
    }

    /// Send a text message; returns its message id. `buttons` attaches an
    /// inline keyboard — rows of `InlineKeyboardButton`s whose presses
    /// arrive as `button` events.
    pub async fn send(
        &self,
        text: &str,
        buttons: Option<InlineKeyboardMarkup>,
    ) -> Result<i64, SenderError> {
        let bubbles = split_bubbles(text);
        let last = bubbles.len() - 1;
        let mut id = 0;
        for (i, bubble) in bubbles.iter().enumerate() {
            self.inter_message_pause().await;
            self.spoke();
            // The keyboard rides the final bubble only.
            let markup = if i == last { buttons.clone() } else { None };
            id = match &self.platform {
                Platform::Telegram(inner) => {
                    inner
                        .client
                        .send_message(
                            inner.chat_id,
                            bubble,
                            self.thread(),
                            markup.map(ReplyMarkup::InlineKeyboard),
                        )
                        .await?
                }
                Platform::Cli(inner) => {
                    let message_id = inner.hub.next_message_id();
                    inner.hub.emit(Outbound::Message(OutboundMessage {
                        chat: inner.chat.clone(),
                        message_id,
                        text: (*bubble).to_string(),
                        buttons: markup_to_buttons(markup.as_ref()),
                        reply_to: None,
                        thread_id: self.thread(),
                        extras: None,
                    }));
                    message_id
                }
                Platform::Discord(inner) => inner
                    .client
                    .send_message_payload(
                        &inner.channel_id,
                        &discord_payload(bubble, markup.as_ref(), None),
                    )
                    .await
                    .map(|m| snowflake_id(&m.id))?,
                #[cfg(test)]
                Platform::Record(out) => {
                    out.send(format!("send:{bubble}"))
                        .await
                        .map_err(|_| SenderError::RecordClosed)?;
                    0
                }
            };
            self.log_outbound(serde_json::json!({
                "type": "message",
                "text": bubble,
                "message_id": id,
            }));
        }
        Ok(id)
    }

    /// Send a text message quoting `message_id`; returns its message id.
    pub async fn reply(
        &self,
        message_id: i64,
        text: &str,
        buttons: Option<InlineKeyboardMarkup>,
    ) -> Result<i64, SenderError> {
        let bubbles = split_bubbles(text);
        let last = bubbles.len() - 1;
        let mut id = 0;
        for (i, bubble) in bubbles.iter().enumerate() {
            self.inter_message_pause().await;
            self.spoke();
            // The quote attaches to the first bubble, the keyboard to the
            // last.
            let markup = if i == last { buttons.clone() } else { None };
            id = match &self.platform {
                Platform::Telegram(inner) => {
                    if i == 0 {
                        inner
                            .client
                            .send_reply_markup(inner.chat_id, message_id, bubble, markup)
                            .await
                            .map_err(|e| bot_err(e, message_id))?
                    } else {
                        inner
                            .client
                            .send_message(
                                inner.chat_id,
                                bubble,
                                self.thread(),
                                markup.map(ReplyMarkup::InlineKeyboard),
                            )
                            .await?
                    }
                }
                Platform::Cli(inner) => {
                    let mid = inner.hub.next_message_id();
                    inner.hub.emit(Outbound::Message(OutboundMessage {
                        chat: inner.chat.clone(),
                        message_id: mid,
                        text: (*bubble).to_string(),
                        buttons: markup_to_buttons(markup.as_ref()),
                        reply_to: (i == 0).then_some(message_id),
                        thread_id: self.thread(),
                        extras: None,
                    }));
                    mid
                }
                Platform::Discord(inner) => inner
                    .client
                    .send_message_payload(
                        &inner.channel_id,
                        &discord_payload(
                            bubble,
                            markup.as_ref(),
                            // The quote attaches to the first bubble.
                            (i == 0).then_some(message_id),
                        ),
                    )
                    .await
                    .map(|m| snowflake_id(&m.id))?,
                #[cfg(test)]
                Platform::Record(out) => {
                    out.send(format!("reply:{message_id}:{bubble}"))
                        .await
                        .map_err(|_| SenderError::RecordClosed)?;
                    0
                }
            };
            self.log_outbound(serde_json::json!({
                "type": "message",
                "text": bubble,
                "message_id": id,
                "reply_to": (i == 0).then_some(message_id),
            }));
        }
        Ok(id)
    }

    /// Send a pack sticker: publish it into the bot's sticker set if the
    /// content is new, then send by `file_id` — a native sticker bubble.
    /// In a private chat the chat id is the user's id, so the caller
    /// supplies it as the set-owner hint.
    pub async fn send_sticker(&self, sticker: &Sticker) -> Result<i64, SenderError> {
        self.inter_message_pause().await;
        self.spoke();
        let id = match &self.platform {
            Platform::Telegram(inner) => {
                // A positive chat id is a private chat whose id equals the
                // user's; group/channel ids are negative.
                let owner_hint = (inner.chat_id > 0).then_some(inner.chat_id);
                let file_id = inner.stickers.publish(sticker, owner_hint).await?;
                inner
                    .client
                    .send_media_id(
                        inner.chat_id,
                        MediaKind::Sticker,
                        &file_id,
                        None,
                        self.thread(),
                    )
                    .await
                    .map_err(SenderError::from)
            }
            Platform::Cli(inner) => {
                // The CLI platform has no sticker store: the sticker file
                // travels as an outbound file the driver can inspect.
                let message_id = inner.hub.next_message_id();
                inner.hub.emit(Outbound::File(OutboundFile {
                    chat: inner.chat.clone(),
                    message_id,
                    kind: "sticker".to_string(),
                    path: Some(sticker.path.display().to_string()),
                    data: None,
                    filename: Some(sticker.name.clone()),
                    caption: None,
                    thread_id: self.thread(),
                }));
                Ok(message_id)
            }
            Platform::Discord(inner) => {
                // No sticker store on Discord: the sticker goes out as a
                // file upload.
                inner
                    .client
                    .send_file(
                        &inner.channel_id,
                        FileSource::Path(sticker.path.clone()),
                        &sticker.name,
                        None,
                    )
                    .await
                    .map(|m| snowflake_id(&m.id))
                    .map_err(SenderError::from)
            }
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("sticker:{}", sticker.name))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(0)
            }
        }?;
        self.log_outbound(serde_json::json!({
            "type": "sticker",
            "name": sticker.name,
            "message_id": id,
        }));
        Ok(id)
    }

    /// Send a sticker Telegram already hosts by `file_id` — for re-sending
    /// stickers that arrived in events.
    pub async fn send_sticker_id(&self, file_id: &str) -> Result<i64, SenderError> {
        self.inter_message_pause().await;
        self.spoke();
        let id = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .send_media_id(
                    inner.chat_id,
                    MediaKind::Sticker,
                    file_id,
                    None,
                    self.thread(),
                )
                .await
                .map_err(SenderError::from),
            Platform::Cli(inner) => {
                // Wire file ids are local paths.
                let message_id = inner.hub.next_message_id();
                inner.hub.emit(Outbound::File(OutboundFile {
                    chat: inner.chat.clone(),
                    message_id,
                    kind: "sticker".to_string(),
                    path: Some(file_id.to_string()),
                    data: None,
                    filename: None,
                    caption: None,
                    thread_id: self.thread(),
                }));
                Ok(message_id)
            }
            // On Discord `file_id` is a CDN URL; re-uploading its bytes is
            // the only way to echo it.
            Platform::Discord(inner) => inner
                .resend_url(file_id, None)
                .await
                .map_err(SenderError::from),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("sticker_id:{file_id}"))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(0)
            }
        }?;
        self.log_outbound(serde_json::json!({
            "type": "sticker",
            "file_id": file_id,
            "message_id": id,
        }));
        Ok(id)
    }

    /// Send a local file as the media kind its extension implies:
    /// images → photo (gif → animation), video → video, audio → audio
    /// (`.ogg` → voice note), anything else → document. Returns the
    /// message id.
    pub async fn send_file(&self, path: &Path, caption: Option<&str>) -> Result<i64, SenderError> {
        self.inter_message_pause().await;
        self.spoke();
        let id = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .send_media(
                    inner.chat_id,
                    media_kind(path),
                    FileSource::Path(path.to_path_buf()),
                    filename(path)?,
                    caption,
                    self.thread(),
                )
                .await
                .map_err(SenderError::from),
            Platform::Cli(inner) => {
                let message_id = inner.hub.next_message_id();
                inner.hub.emit(Outbound::File(OutboundFile {
                    chat: inner.chat.clone(),
                    message_id,
                    kind: kind_str(media_kind(path)).to_string(),
                    path: Some(path.display().to_string()),
                    data: None,
                    filename: Some(filename(path)?.to_string()),
                    caption: caption.map(str::to_string),
                    thread_id: self.thread(),
                }));
                Ok(message_id)
            }
            Platform::Discord(inner) => inner
                .client
                .send_file(
                    &inner.channel_id,
                    FileSource::Path(path.to_path_buf()),
                    filename(path)?,
                    caption,
                )
                .await
                .map(|m| snowflake_id(&m.id))
                .map_err(SenderError::from),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("file:{}", path.display()))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(0)
            }
        }?;
        self.log_outbound(serde_json::json!({
            "type": "file",
            "path": path.display().to_string(),
            "caption": caption,
            "message_id": id,
        }));
        Ok(id)
    }

    /// Re-send media Telegram already hosts by `file_id`, as `kind` — for
    /// echoing back media from inbound events.
    pub async fn send_file_id(
        &self,
        kind: MediaKind,
        file_id: &str,
        caption: Option<&str>,
    ) -> Result<i64, SenderError> {
        self.inter_message_pause().await;
        self.spoke();
        let id = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .send_media_id(inner.chat_id, kind, file_id, caption, self.thread())
                .await
                .map_err(SenderError::from),
            Platform::Cli(inner) => {
                let message_id = inner.hub.next_message_id();
                inner.hub.emit(Outbound::File(OutboundFile {
                    chat: inner.chat.clone(),
                    message_id,
                    kind: kind_str(kind).to_string(),
                    path: Some(file_id.to_string()),
                    data: None,
                    filename: None,
                    caption: caption.map(str::to_string),
                    thread_id: self.thread(),
                }));
                Ok(message_id)
            }
            // On Discord `file_id` is a CDN URL; re-uploading its bytes is
            // the only way to echo it.
            Platform::Discord(inner) => inner
                .resend_url(file_id, caption)
                .await
                .map_err(SenderError::from),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("file_id:{kind:?}:{file_id}"))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(0)
            }
        }?;
        self.log_outbound(serde_json::json!({
            "type": "file",
            "kind": kind_str(kind),
            "file_id": file_id,
            "caption": caption,
            "message_id": id,
        }));
        Ok(id)
    }

    /// Download the platform file `file_id` points to into `inbox/` inside
    /// `chat_dir`, named by the id so repeat events hit the disk cache.
    /// Returns the path relative to `chat_dir` plus the guessed MIME type —
    /// `None` when no local copy can be produced (Telegram's `getFile`
    /// refuses expired or >20MB files; a CLI path may not exist). The event
    /// still arrives, just without a local file.
    pub async fn fetch_media(
        &self,
        file_id: &str,
        chat_dir: &Path,
    ) -> Result<Option<EventFile>, SenderError> {
        let inbox = chat_dir.join("inbox");
        let relative = |abs: &Path| EventFile {
            path: abs
                .strip_prefix(chat_dir)
                .expect("fetched file lives under chat_dir")
                .to_string_lossy()
                .into_owned(),
            mime: mime_guess::from_path(abs)
                .first_or_octet_stream()
                .to_string(),
        };
        match &self.platform {
            Platform::Telegram(inner) => {
                let safe_id: String = file_id
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                // A previously downloaded copy skips `getFile` entirely.
                let cached = std::fs::read_dir(&inbox)
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(std::result::Result::ok)
                    .find(|entry| {
                        entry
                            .file_name()
                            .to_str()
                            .is_some_and(|name| name.starts_with(&format!("{safe_id}.")))
                    })
                    .map(|entry| entry.path());
                if let Some(abs) = cached {
                    return Ok(Some(relative(&abs)));
                }
                let file = match inner.client.get_file(file_id).await {
                    Ok(file) => file,
                    Err(error) => {
                        warn!(%error, "getFile failed; event arrives without local copy");
                        return Ok(None);
                    }
                };
                let Some(file_path) = file.file_path else {
                    return Ok(None);
                };
                let ext = Path::new(&file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("bin");
                let rel = Path::new("inbox").join(format!("{safe_id}.{ext}"));
                let abs = chat_dir.join(&rel);
                let bytes = match inner.client.download_file(&file_path, DOWNLOAD_LIMIT).await {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        warn!(%error, "media download failed; event arrives without local copy");
                        return Ok(None);
                    }
                };
                async_fs::create_dir_all(abs.parent().expect("inbox/… has a parent")).await?;
                async_fs::write(&abs, &bytes).await?;
                Ok(Some(relative(&abs)))
            }
            Platform::Cli(_) => {
                // The wire "file id" is a local path. Copy it into inbox/
                // so the agent can read it — sandboxed agents cannot open
                // paths outside the chat dir.
                let src = Path::new(file_id);
                let Some(name) = src.file_name().and_then(|n| n.to_str()) else {
                    return Ok(None);
                };
                let abs = inbox.join(name);
                if !abs.exists() {
                    if let Err(error) = async_fs::create_dir_all(&inbox).await {
                        warn!(%error, "inbox mkdir failed; event arrives without local copy");
                        return Ok(None);
                    }
                    if let Err(error) = async_fs::copy(src, &abs).await {
                        warn!(%error, src = %src.display(),
                              "media copy failed; event arrives without local copy");
                        return Ok(None);
                    }
                }
                Ok(Some(relative(&abs)))
            }
            Platform::Discord(inner) => {
                // `file_id` is the attachment's CDN URL; its filename is
                // the last path segment (query params stripped — CDN urls
                // are signed).
                let Some(name) = attachment_name(file_id) else {
                    return Ok(None);
                };
                let abs = inbox.join(name);
                if !abs.exists() {
                    let bytes = match inner.client.download(file_id, DOWNLOAD_LIMIT).await {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            warn!(%error, "attachment download failed; event arrives without local copy");
                            return Ok(None);
                        }
                    };
                    async_fs::create_dir_all(&inbox).await?;
                    async_fs::write(&abs, &bytes).await?;
                }
                Ok(Some(relative(&abs)))
            }
            #[cfg(test)]
            Platform::Record(_) => Ok(None),
        }
    }

    /// Set the bot's emoji reaction on `message_id`; `None` clears it.
    /// `is_big` plays the large animation.
    pub async fn react(
        &self,
        message_id: i64,
        emoji: Option<&str>,
        is_big: bool,
    ) -> Result<(), SenderError> {
        self.spoke();
        let result = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .set_message_reaction(inner.chat_id, message_id, emoji, is_big)
                .await
                .map_err(|e| bot_err(e, message_id)),
            Platform::Cli(inner) => {
                inner.hub.emit(Outbound::Reaction(OutboundReaction {
                    chat: inner.chat.clone(),
                    message_id,
                    emojis: emoji.map_or_else(Vec::new, |e| vec![e.to_string()]),
                    is_big,
                }));
                Ok(())
            }
            Platform::Discord(inner) => {
                let message_id = message_id.to_string();
                match emoji {
                    // `is_big` has no Discord equivalent.
                    Some(emoji) => inner
                        .client
                        .react(&inner.channel_id, &message_id, emoji)
                        .await
                        .map_err(|e| bot_err(e, snowflake_id(&message_id))),
                    // Discord removes the bot's reaction per emoji, not
                    // wholesale — without an emoji there is nothing to
                    // clear.
                    None => Ok(()),
                }
            }
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("react:{message_id}:{}", emoji.unwrap_or("")))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(())
            }
        };
        if result.is_ok() {
            self.log_outbound(serde_json::json!({
                "type": "reaction",
                "message_id": message_id,
                "emoji": emoji,
            }));
        }
        result
    }

    /// Edit the text of a message the bot sent.
    pub async fn edit(&self, message_id: i64, text: &str) -> Result<(), SenderError> {
        let result = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .edit_message_text(inner.chat_id, message_id, text, None)
                .await
                .map_err(|e| bot_err(e, message_id)),
            Platform::Cli(inner) => {
                inner.hub.emit(Outbound::Edit(OutboundEdit {
                    chat: inner.chat.clone(),
                    message_id,
                    text: text.to_string(),
                }));
                Ok(())
            }
            Platform::Discord(inner) => inner
                .client
                .edit_message(&inner.channel_id, &message_id.to_string(), text)
                .await
                .map(|_| ())
                .map_err(|e| bot_err(e, message_id)),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("edit:{message_id}:{text}"))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(())
            }
        };
        if result.is_ok() {
            self.log_outbound(serde_json::json!({
                "type": "edit",
                "message_id": message_id,
                "text": text,
            }));
        }
        result
    }

    /// Delete a message (the bot's own, or any where it has delete rights).
    pub async fn delete_message(&self, message_id: i64) -> Result<(), SenderError> {
        let result = match &self.platform {
            Platform::Telegram(inner) => inner
                .client
                .delete_message(inner.chat_id, message_id)
                .await
                .map_err(|e| bot_err(e, message_id)),
            Platform::Cli(inner) => {
                inner.hub.emit(Outbound::Delete(OutboundDelete {
                    chat: inner.chat.clone(),
                    message_id,
                }));
                Ok(())
            }
            Platform::Discord(inner) => inner
                .client
                .delete_message(&inner.channel_id, &message_id.to_string())
                .await
                .map_err(|e| bot_err(e, message_id)),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("delete:{message_id}"))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(())
            }
        };
        if result.is_ok() {
            self.log_outbound(serde_json::json!({
                "type": "delete",
                "message_id": message_id,
            }));
        }
        result
    }

    /// Pin or unpin `message_id`. `notify = false` pins silently.
    pub async fn pin(&self, message_id: i64, unpin: bool, notify: bool) -> Result<(), SenderError> {
        let result = match &self.platform {
            Platform::Telegram(inner) => if unpin {
                inner.client.unpin_message(inner.chat_id, message_id).await
            } else {
                inner
                    .client
                    .pin_message(inner.chat_id, message_id, notify)
                    .await
            }
            .map_err(|e| bot_err(e, message_id)),
            Platform::Cli(inner) => {
                inner.hub.emit(Outbound::Pin(OutboundPin {
                    chat: inner.chat.clone(),
                    message_id,
                    unpin,
                    notify,
                }));
                Ok(())
            }
            Platform::Discord(inner) => inner
                .client
                // `notify` has no Discord equivalent — pins always post a
                // system message.
                .pin_message(&inner.channel_id, &message_id.to_string(), unpin)
                .await
                .map_err(|e| bot_err(e, message_id)),
            #[cfg(test)]
            Platform::Record(out) => {
                out.send(format!("pin:{message_id}:{unpin}"))
                    .await
                    .map_err(|_| SenderError::RecordClosed)?;
                Ok(())
            }
        };
        if result.is_ok() {
            self.log_outbound(serde_json::json!({
                "type": if unpin { "unpin" } else { "pin" },
                "message_id": message_id,
            }));
        }
        result
    }

    /// Whether `message_id` still exists in the chat — `false` means it
    /// was deleted. Telegram pushes no deletion update, so existence is
    /// probed: a no-op `editMessageReplyMarkup` answers "not modified" on
    /// a live message and "message to edit not found" on a dead one.
    pub async fn probe_message(&self, message_id: i64) -> Result<bool, SenderError> {
        match &self.platform {
            Platform::Telegram(inner) => match inner
                .client
                .edit_message_reply_markup(inner.chat_id, message_id, None)
                .await
            {
                Ok(()) => Ok(true),
                Err(error) if message_gone(&error) => Ok(false),
                // "not modified" also means the message is there.
                Err(BotError::Api(desc)) if desc.contains("not modified") => Ok(true),
                Err(error) => Err(SenderError::Platform(error)),
            },
            // A deleted message answers 404 to a plain GET.
            Platform::Discord(inner) => inner
                .client
                .get_message(&inner.channel_id, &message_id.to_string())
                .await
                .map(|message| message.is_some())
                .map_err(|e| bot_err(e, message_id)),
            // The CLI/test platforms have no deletable message store.
            Platform::Cli(_) => Ok(true),
            #[cfg(test)]
            Platform::Record(_) => Ok(true),
        }
    }

    /// Look up a chat the bot can see — `query` is a numeric id, an
    /// `@username`, or a `t.me` link. Returns the public metadata plus the
    /// bot's own membership status; a chat the bot cannot reach answers
    /// the platform's error.
    pub async fn chat_info(&self, query: &str) -> Result<serde_json::Value, SenderError> {
        match &self.platform {
            Platform::Telegram(inner) => {
                let (chat, _message) = telegram_ref(query)?;
                let info = inner.client.get_chat(chat.clone()).await?;
                let members = inner.client.get_chat_member_count(chat.clone()).await.ok();
                let status = inner
                    .client
                    .get_chat_member(chat, inner.bot_id)
                    .await
                    .map(|member| member.status)
                    .unwrap_or_else(|_| "not a member".to_string());
                // Only chats the bot sits in push it events (or answer
                // fetch_message) — `restricted`/`left`/`kicked` don't.
                let readable = matches!(status.as_str(), "creator" | "administrator" | "member");
                Ok(serde_json::json!({
                    "id": info.id,
                    "type": format!("{:?}", info.chat_type).to_lowercase(),
                    "title": info.title,
                    "username": info.username,
                    "first_name": info.first_name,
                    "description": info.description,
                    "bio": info.bio,
                    "invite_link": info.invite_link,
                    "linked_chat_id": info.linked_chat_id,
                    "has_visible_history": info.has_visible_history,
                    "members": members,
                    "bot_status": status,
                    "readable": readable,
                }))
            }
            _ => Err(SenderError::Unsupported("chat_info")),
        }
    }

    /// Read one message out of a chat the bot belongs to: `forwardMessage`
    /// lands a copy in this sender's chat whose response carries the
    /// content; the copy is deleted right after. `query` accepts the same
    /// forms as [`Self::chat_info`] — a message link (`t.me/<c>/<id>`)
    /// supplies `message_id` itself.
    pub async fn fetch_message(
        &self,
        query: &str,
        message_id: Option<i64>,
    ) -> Result<serde_json::Value, SenderError> {
        match &self.platform {
            Platform::Telegram(inner) => {
                let (chat, linked) = telegram_ref(query)?;
                let message_id = message_id.or(linked).ok_or(SenderError::MissingMessageId)?;
                let message = inner
                    .client
                    .forward_message(inner.chat_id, chat, message_id)
                    .await?;
                // The forwarded copy was only a way to read the source —
                // remove it so nothing user-visible is left behind.
                let _ = inner
                    .client
                    .delete_message(inner.chat_id, message.message_id)
                    .await;
                let media = if message.sticker.is_some() {
                    Some("sticker")
                } else if message.photo.is_some() {
                    Some("photo")
                } else if message.animation.is_some() {
                    Some("animation")
                } else if message.video.is_some() {
                    Some("video")
                } else if message.audio.is_some() {
                    Some("audio")
                } else if message.voice.is_some() {
                    Some("voice")
                } else if message.document.is_some() {
                    Some("document")
                } else {
                    None
                };
                // The copy's own `from` is the bot; the original author
                // lives on `forward_origin`.
                let from = message.forward_origin.and_then(|origin| {
                    origin
                        .sender_user
                        .map(|u| u.username.unwrap_or(u.first_name))
                        .or(origin.sender_user_name)
                        .or_else(|| origin.chat.and_then(|c| c.title.or(c.username)))
                });
                Ok(serde_json::json!({
                    "message_id": message_id,
                    "date": message.date,
                    "from": from,
                    "text": message.text,
                    "caption": message.caption,
                    "media": media,
                }))
            }
            _ => Err(SenderError::Unsupported("fetch_message")),
        }
    }

    /// Show "typing…" in the chat until the returned guard is dropped.
    pub fn typing_guard(&self) -> ChatActionGuard {
        match &self.platform {
            Platform::Telegram(inner) => ChatActionGuard::start(
                botkit_core::action::AnyChatActionSender::new(TelegramActionSender::new(
                    inner.client.clone(),
                    inner.chat_id,
                    self.thread(),
                )),
                ChatAction::Typing,
            ),
            Platform::Cli(inner) => ChatActionGuard::start(
                botkit_core::action::AnyChatActionSender::new(CliActionSender::new(
                    inner.hub.clone(),
                    inner.chat.clone(),
                    self.thread(),
                )),
                ChatAction::Typing,
            ),
            Platform::Discord(inner) => ChatActionGuard::start(
                botkit_core::action::AnyChatActionSender::new(DiscordActionSender::new(
                    inner.client.clone(),
                    inner.channel_id.clone(),
                )),
                ChatAction::Typing,
            ),
            #[cfg(test)]
            Platform::Record(_) => ChatActionGuard::start(
                botkit_core::action::AnyChatActionSender::new(NoopAction),
                ChatAction::Typing,
            ),
        }
    }
}

impl DiscordSender {
    /// Re-send an attachment Discord already hosts: `file_id` is its CDN
    /// url, so the bytes come down and go back up as an upload.
    async fn resend_url(&self, url: &str, caption: Option<&str>) -> Result<i64, BotError> {
        let bytes = self.client.download(url, DOWNLOAD_LIMIT).await?;
        let name = attachment_name(url).unwrap_or("file");
        self.client
            .send_file(&self.channel_id, FileSource::Bytes(bytes), name, caption)
            .await
            .map(|m| snowflake_id(&m.id))
    }
}

/// Parse the `chat` argument of the lookup tools: a numeric id, an
/// `@username` (a bare word counts), or a `t.me`/`telegram.me` link. A
/// message link (`t.me/<name>/<msg>`, `t.me/c/<internal>/<msg>`) also
/// yields its message id; `t.me/c/<internal>` maps to the `-100…` chat id
/// form the API uses. Invite links (`t.me/+…`, `joinchat`) cannot resolve
/// — the Bot API has no invite lookup.
fn telegram_ref(raw: &str) -> Result<(ChatRef, Option<i64>), SenderError> {
    let bad = || SenderError::BadChatRef(raw.to_string());
    let mut s = raw.trim().strip_prefix("telegram:").unwrap_or(raw.trim());
    for prefix in ["https://", "http://"] {
        s = s.strip_prefix(prefix).unwrap_or(s);
    }
    for host in ["t.me/", "telegram.me/", "telegram.dog/"] {
        s = s.strip_prefix(host).unwrap_or(s);
    }
    // `t.me/s/<name>` is the web-preview form of `t.me/<name>`.
    s = s.strip_prefix("s/").unwrap_or(s);
    if let Ok(id) = s.parse::<i64>() {
        return Ok((ChatRef::Id(id), None));
    }
    let mut segments = s.split('/');
    let head = segments.next().unwrap_or_default();
    if head == "c" {
        // Private-link form: the internal channel id becomes -100…
        return segments
            .next()
            .and_then(|id| id.parse::<i64>().ok())
            .map(|internal| {
                (
                    ChatRef::Id(-(1_000_000_000_000 + internal)),
                    segments.next().and_then(|msg| msg.parse::<i64>().ok()),
                )
            })
            .ok_or_else(bad);
    }
    let name = head.strip_prefix('@').unwrap_or(head);
    if name.is_empty()
        || name.starts_with('+')
        || name == "joinchat"
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(bad());
    }
    let message_id = segments.next().and_then(|msg| msg.parse::<i64>().ok());
    Ok((ChatRef::Username(format!("@{name}")), message_id))
}

/// A Discord snowflake id as the `i64` ids the tool layer uses. Real
/// snowflakes sit far below `i64::MAX`; saturating keeps a hypothetical
/// overflow representable instead of panicking.
pub(crate) fn snowflake_id(id: &str) -> i64 {
    id.parse::<u64>()
        .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// The filename an attachment URL ends in — last path segment, query
/// params stripped (CDN urls are signed).
fn attachment_name(url: &str) -> Option<&str> {
    let path = url.split('?').next()?;
    path.rsplit('/').next().filter(|name| !name.is_empty())
}

/// The Discord channel-message payload for one bubble: `content`, button
/// rows as components, and a `message_reference` when it quotes another
/// message.
fn discord_payload(
    text: &str,
    markup: Option<&InlineKeyboardMarkup>,
    reply_to: Option<i64>,
) -> serde_json::Value {
    let mut payload = serde_json::json!({ "content": text });
    if let Some(components) = markup_to_components(markup) {
        payload["components"] = components;
    }
    if let Some(reply_to) = reply_to {
        payload["message_reference"] = serde_json::json!({
            "message_id": reply_to.to_string(),
            // A quote of a since-deleted message sends as a plain message.
            "fail_if_not_exists": false,
        });
    }
    payload
}

/// Telegram-style button rows rendered as Discord action rows: a button
/// with a `url` becomes a link button (style 5), anything else a primary
/// button (style 1) keyed by `custom_id`.
fn markup_to_components(markup: Option<&InlineKeyboardMarkup>) -> Option<serde_json::Value> {
    let rows: Vec<serde_json::Value> = markup?
        .inline_keyboard
        .iter()
        .map(|row| {
            serde_json::json!({
                "type": 1,
                "components": row
                    .iter()
                    .map(|button| match &button.url {
                        Some(url) => serde_json::json!({
                            "type": 2,
                            "style": 5,
                            "label": button.text,
                            "url": url,
                        }),
                        None => serde_json::json!({
                            "type": 2,
                            "style": 1,
                            "label": button.text,
                            // custom_id is mandatory for non-link buttons;
                            // fall back to the label when no data was set.
                            "custom_id": button
                                .callback_data
                                .clone()
                                .unwrap_or_else(|| button.text.clone()),
                        }),
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    (!rows.is_empty()).then_some(serde_json::Value::Array(rows))
}

/// Whether the API error means the message no longer exists — Telegram's
/// "Bad Request: message to <verb> not found" variants and Discord's
/// "Unknown Message" 404.
fn message_gone(error: &BotError) -> bool {
    matches!(error, BotError::Api(desc) if
        (desc.contains("message") && desc.contains("not found"))
            || desc.contains("Unknown Message"))
}

/// Translate a platform error on `message_id`: a `not found` reply means
/// the message was deleted — say so plainly rather than relaying the raw
/// API description.
fn bot_err(error: BotError, message_id: i64) -> SenderError {
    if message_gone(&error) {
        SenderError::MessageGone(message_id)
    } else {
        SenderError::Platform(error)
    }
}

/// The media kind a file's MIME type maps to.
fn media_kind(path: &Path) -> MediaKind {
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    match (mime.type_(), mime.subtype().as_str()) {
        (mime_guess::mime::IMAGE, "gif") => MediaKind::Animation,
        (mime_guess::mime::IMAGE, _) => MediaKind::Photo,
        (mime_guess::mime::VIDEO, _) => MediaKind::Video,
        (mime_guess::mime::AUDIO, "ogg") => MediaKind::Voice,
        (mime_guess::mime::AUDIO, _) => MediaKind::Audio,
        _ => MediaKind::Document,
    }
}

/// A `ChatActionSender` that never touches a platform API (tests only).
#[cfg(test)]
struct NoopAction;

#[cfg(test)]
impl botkit_core::action::ChatActionSender for NoopAction {
    fn send_action(
        &self,
        _: ChatAction,
    ) -> impl botkit_core::action::ChatActionFutureBounds<Output = Result<(), botkit_core::BotError>> + '_
    {
        async { Ok(()) }
    }

    fn action_expiry(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

fn filename(path: &Path) -> Result<&str, SenderError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| SenderError::NoFilename(path.to_path_buf()))
}

/// `MediaKind` as the wire-protocol kind string.
fn kind_str(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Photo => "photo",
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
        MediaKind::Voice => "voice",
        MediaKind::Animation => "animation",
        MediaKind::Sticker => "sticker",
        MediaKind::Document => "document",
    }
}

/// Convert an `InlineKeyboardMarkup` into CLI keyboard rows.
fn markup_to_buttons(markup: Option<&InlineKeyboardMarkup>) -> Vec<Vec<WireButton>> {
    markup
        .map(|markup| {
            markup
                .inline_keyboard
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|button| WireButton {
                            text: button.text.clone(),
                            data: button.callback_data.clone(),
                            url: button.url.clone(),
                        })
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_kind_follows_extension() {
        let cases = [
            ("a.png", MediaKind::Photo),
            ("a.jpg", MediaKind::Photo),
            ("a.gif", MediaKind::Animation),
            ("a.mp4", MediaKind::Video),
            ("a.mp3", MediaKind::Audio),
            ("a.flac", MediaKind::Audio),
            ("a.ogg", MediaKind::Voice),
            ("a.pdf", MediaKind::Document),
            ("a.zip", MediaKind::Document),
        ];
        for (name, expected) in cases {
            assert_eq!(media_kind(Path::new(name)), expected, "{name}");
        }
    }

    #[test]
    fn short_text_stays_one_bubble() {
        assert_eq!(split_bubbles("午饭吃什么😋"), vec!["午饭吃什么😋"]);
        assert_eq!(
            split_bubbles("two lines\nbut short"),
            vec!["two lines\nbut short"]
        );
    }

    #[test]
    fn blank_lines_split_into_bubbles() {
        assert_eq!(
            split_bubbles("first thought\n\nsecond thought\n\n\nthird"),
            vec!["first thought", "second thought", "third"]
        );
    }

    #[test]
    fn overlong_paragraph_splits_at_lines_then_sentences() {
        // Single newlines split when the paragraph exceeds the cap.
        let long = format!("{}\n{}", "甲".repeat(200), "乙".repeat(100));
        let bubbles = split_bubbles(&long);
        assert_eq!(bubbles.len(), 2);
        assert!(bubbles[0].starts_with('甲') && bubbles[1].starts_with('乙'));

        // One giant line falls back to sentence boundaries.
        let blob = format!(
            "{}。{}！{}",
            "甲".repeat(150),
            "乙".repeat(150),
            "丙".repeat(10)
        );
        let bubbles = split_bubbles(&blob);
        assert_eq!(bubbles.len(), 3);
        assert!(bubbles[0].ends_with('。') && bubbles[1].ends_with('！'));
    }

    #[test]
    fn period_only_splits_before_whitespace() {
        let text = "Done already. Now checking what broke in the failing test \
                    suite on v1.2 of the build. Will report back.";
        let bubbles = split_bubbles(text);
        assert_eq!(
            bubbles,
            vec![
                "Done already.",
                "Now checking what broke in the failing test suite on v1.2 of the \
                 build.",
                "Will report back."
            ]
        );
    }

    #[test]
    fn fenced_code_is_never_split() {
        let code = format!("```rust\n{}\n```", "x = 1;\n".repeat(80));
        let text = format!("here is the code\n\n{code}\n\nand that is it");
        let bubbles = split_bubbles(&text);
        assert_eq!(bubbles.len(), 3);
        assert!(bubbles[1].starts_with("```rust") && bubbles[1].ends_with("```"));
    }

    #[test]
    fn whitespace_only_falls_back_to_trimmed_text() {
        assert_eq!(split_bubbles("  \n\n  "), vec![""]);
    }

    /// Telegram's "message to <verb> not found" is a deletion; unrelated
    /// API failures stay platform errors.
    #[test]
    fn gone_errors_translate() {
        assert!(message_gone(&BotError::Api(
            "Bad Request: message to react not found".into()
        )));
        assert!(message_gone(&BotError::Api(
            "Bad Request: message to edit not found".into()
        )));
        assert!(!message_gone(&BotError::Api(
            "Bad Request: chat not found".into()
        )));
        assert!(!message_gone(&BotError::Api(
            "Forbidden: bot was blocked by the user".into()
        )));

        let gone = BotError::Api("Bad Request: message to delete not found".into());
        assert!(matches!(bot_err(gone, 7), SenderError::MessageGone(7)));
        let other = BotError::Api("Bad Request: chat not found".into());
        assert!(matches!(
            bot_err(other, 7),
            SenderError::Platform(BotError::Api(_))
        ));
    }

    /// Live Telegram probe: send → exists → delete → gone.
    ///
    /// ```sh
    /// TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT=12345 \
    ///     cargo test probe_real_message -- --ignored
    /// ```
    #[test]
    #[ignore = "needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT and network access"]
    fn probe_real_message() {
        let token = std::env::var("TELEGRAM_BOT_TOKEN").expect("TELEGRAM_BOT_TOKEN");
        let chat = std::env::var("TELEGRAM_CHAT").expect("TELEGRAM_CHAT");
        let client = TelegramClient::new(token);
        let dir = std::env::temp_dir().join(format!("acpbot-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (spoke_tx, _spoke_rx) = async_channel::unbounded();
        let sender = Sender::telegram(
            client,
            &ChatKey {
                platform: "telegram",
                id: chat,
            },
            spoke_tx,
            Arc::new(AtomicI64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(crate::stickerset::StickerSet::new(
                TelegramClient::new(std::env::var("TELEGRAM_BOT_TOKEN").unwrap()),
                dir.clone(),
                None,
                None,
                None,
            )),
            0,
        )
        .expect("telegram sender");

        let executor = async_executor::Executor::new();
        futures_lite::future::block_on(executor.run(async {
            let id = sender.send("probe target", None).await.expect("send");
            assert!(sender.probe_message(id).await.expect("probe live"));
            sender.delete_message(id).await.expect("delete");
            assert!(matches!(sender.probe_message(id).await, Ok(false)));
            // Probing a never-existent id is `false` too, not an error.
            assert!(matches!(
                sender.probe_message(id + 100_000).await,
                Ok(false)
            ));
        }));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The `chat` argument's many spellings all land on one `ChatRef`.
    #[test]
    fn telegram_ref_forms() {
        let id = |raw| match telegram_ref(raw) {
            Ok((ChatRef::Id(id), msg)) => (id, msg),
            other => panic!("{raw} → {other:?}"),
        };
        let name = |raw| match telegram_ref(raw) {
            Ok((ChatRef::Username(name), msg)) => (name, msg),
            other => panic!("{raw} → {other:?}"),
        };

        assert_eq!(id("-1002495551562"), (-1002495551562, None));
        assert_eq!(id("telegram:-1002495551562"), (-1002495551562, None));
        assert_eq!(id("t.me/c/2495551562/42"), (-1002495551562, Some(42)));
        assert_eq!(id("https://t.me/c/2495551562"), (-1002495551562, None));
        assert_eq!(name("@durov"), ("@durov".to_string(), None));
        assert_eq!(name("durov"), ("@durov".to_string(), None));
        assert_eq!(name("t.me/durov/42"), ("@durov".to_string(), Some(42)));
        assert_eq!(
            name("https://t.me/s/durov/42"),
            ("@durov".to_string(), Some(42))
        );

        for bad in [
            "t.me/+AbCdEf",
            "https://t.me/joinchat/AbCdEf",
            "not a chat",
            "",
        ] {
            assert!(
                matches!(telegram_ref(bad), Err(SenderError::BadChatRef(_))),
                "{bad}"
            );
        }
    }
}
