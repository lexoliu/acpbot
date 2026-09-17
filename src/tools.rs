//! MCP tools the agent uses to speak in the chat — and to evolve itself.
//!
//! One [`Tools`] registry is built per accepted bridge connection. The
//! single agent session serves every chat, so chat-bound tools resolve
//! their target through the shared [`ChatRouter`]: an optional `chat`
//! argument picks a conversation, and its absence means the chat whose
//! events triggered the in-flight turn. The daemon serves it through
//! `aither_mcp::McpServer`; the agent sees it as MCP server `chat`.

use std::path::PathBuf;

use aither_core::llm::tool::{Tool, ToolResult, Tools};
use async_channel::Sender as ChanSender;
use botkit_telegram::{InlineKeyboardButton, InlineKeyboardMarkup, MediaKind};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::agent::ChatRouter;
use crate::history::History;
use crate::sender::Sender;
use crate::stickerlib::{LibrarySticker, StickerLibrary};
use crate::stickers::StickerPack;
use std::sync::Arc;

/// Build the chat toolset, backed by the shared chat router.
///
/// `router`'s sticker directory is re-scanned on every sticker call so a
/// file the agent just dropped into the pack is immediately sendable;
/// `restart` receives a unit when the agent asks to be reincarnated.
pub fn chat_tools(router: Arc<ChatRouter>, restart: ChanSender<()>) -> Tools {
    let sticker_dir = router.shared().sticker_dir.clone();
    let library = router.shared().sticker_library.clone();
    let mut tools = Tools::new();
    register(
        &mut tools,
        SendMessage {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        Reply {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        SendSticker {
            router: router.clone(),
            sticker_dir: sticker_dir.clone(),
            library: library.clone(),
        },
    );
    register(
        &mut tools,
        SendFile {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        React {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        EditMessage {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        DeleteMessage {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        PinMessage {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        MessageStatus {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        ListStickers {
            sticker_dir,
            library: library.clone(),
        },
    );
    register(
        &mut tools,
        ImportStickerSet {
            library: library.clone(),
        },
    );
    register(
        &mut tools,
        SaveSticker {
            library: library.clone(),
        },
    );
    register(&mut tools, ListStickerSets { library });
    register(
        &mut tools,
        ChatInfo {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        FetchMessage {
            router: router.clone(),
        },
    );
    register(
        &mut tools,
        ChatHistory {
            router: router.clone(),
        },
    );
    register(&mut tools, SearchHistory { router });
    register(&mut tools, Restart { restart });
    tools
}

fn register(tools: &mut Tools, tool: impl Tool + 'static) {
    tools
        .register(tool)
        .expect("chat tool registration is static and cannot fail");
}

/// Resolve a call's `chat` argument to its [`Sender`]. Failures are usage
/// errors surfaced as tool results, not protocol errors.
fn target(router: &ChatRouter, chat: Option<&str>) -> Result<Sender, ToolResult> {
    router.sender_for(chat).map_err(ToolResult::error)
}

/// The pack is re-read from disk so agents can extend it mid-session.
fn load_pack(dir: &std::path::Path) -> aither_core::Result<StickerPack> {
    StickerPack::load(dir)
        .map_err(|e| aither_core::Error::msg(format!("sticker pack failed to load: {e}")))
}

/// Send a new text message to the chat.
struct SendMessage {
    router: Arc<ChatRouter>,
}

/// Arguments for `send_message`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendMessageArgs {
    /// The message text to send.
    text: String,
    /// Optional inline keyboard: rows of buttons, each `{text, data}`
    /// (a press arrives as a `button` event) or `{text, url}` (a link).
    buttons: Option<Vec<Vec<ButtonArg>>>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

/// One inline-keyboard button: a callback (`data`) or a link (`url`).
#[derive(Debug, Deserialize, JsonSchema)]
struct ButtonArg {
    /// The label on the button.
    text: String,
    /// Callback payload — the `button` field of the event a press produces.
    data: Option<String>,
    /// A URL the button opens.
    url: Option<String>,
}

/// Build the Telegram markup for `buttons` args shared by `send_message`
/// and `reply`; `None`/empty stays `None`. An invalid button is a usage
/// error message, not a panic.
fn keyboard(buttons: Option<Vec<Vec<ButtonArg>>>) -> Result<Option<InlineKeyboardMarkup>, String> {
    let Some(rows) = buttons else {
        return Ok(None);
    };
    let mut markup = Vec::with_capacity(rows.len());
    for row in rows {
        let mut buttons = Vec::with_capacity(row.len());
        for button in row {
            let ButtonArg { text, data, url } = button;
            buttons.push(match (data, url) {
                (Some(data), None) => InlineKeyboardButton::callback(text, data),
                (None, Some(url)) => InlineKeyboardButton::url(text, url),
                _ => {
                    return Err("each button needs exactly one of `data` or `url`".to_string());
                }
            });
        }
        markup.push(buttons);
    }
    if markup.iter().all(Vec::is_empty) {
        return Ok(None);
    }
    Ok(Some(InlineKeyboardMarkup {
        inline_keyboard: markup,
    }))
}

impl Tool for SendMessage {
    type Arguments = SendMessageArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "send_message".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Send a text message to the chat. This is the only way to say something \
         to the users — text written outside tool calls is never delivered. \
         Speak like a person: several short calls, one thought each, instead \
         of one long message — each call is a separate bubble. `buttons` \
         attaches an inline keyboard: rows of `{text, data}` (a press arrives \
         as a `button` event) or `{text, url}` links."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let markup = match keyboard(args.buttons) {
            Ok(markup) => markup,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        let id = sender.send(&args.text, markup).await?;
        Ok(ToolResult::text(format!("sent message {id}")))
    }
}

/// Reply to a specific message.
struct Reply {
    router: Arc<ChatRouter>,
}

/// Arguments for `reply`.
#[derive(Debug, Deserialize, JsonSchema)]
struct ReplyArgs {
    /// The `message_id` of the message to reply to, from the event JSON.
    message_id: i64,
    /// The reply text.
    text: String,
    /// Optional inline keyboard — same shape as `send_message`'s `buttons`.
    buttons: Option<Vec<Vec<ButtonArg>>>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for Reply {
    type Arguments = ReplyArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "reply".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Send a message that quotes a specific earlier message. Every incoming \
         event carries its `message_id`. `buttons` attaches an inline \
         keyboard like `send_message`'s."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let markup = match keyboard(args.buttons) {
            Ok(markup) => markup,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        let id = sender.reply(args.message_id, &args.text, markup).await?;
        Ok(ToolResult::text(format!("sent reply {id}")))
    }
}

/// Send a sticker — from the pack by name, or by Telegram `file_id`.
struct SendSticker {
    router: Arc<ChatRouter>,
    sticker_dir: PathBuf,
    library: Arc<StickerLibrary>,
}

/// Arguments for `send_sticker`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendStickerArgs {
    /// Sticker name as returned by `list_stickers`.
    name: Option<String>,
    /// A Telegram `file_id` — e.g. to resend a sticker an event carried.
    file_id: Option<String>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for SendSticker {
    type Arguments = SendStickerArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "send_sticker".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Send a native Telegram sticker. Give `name` for a pack or library \
         sticker (`list_stickers` shows every available name) — pack \
         stickers publish into the bot's sticker set on first send, library \
         names resolve to a saved `file_id` — or `file_id` to resend a \
         sticker an event carried."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        if let Some(file_id) = args.file_id {
            let id = sender.send_sticker_id(&file_id).await?;
            return Ok(ToolResult::text(format!("sent sticker as message {id}")));
        }
        let Some(name) = args.name else {
            return Ok(ToolResult::error("give `name` or `file_id`"));
        };
        let pack = load_pack(&self.sticker_dir)?;
        if let Some(sticker) = pack.get(&name) {
            let id = sender.send_sticker(sticker).await?;
            return Ok(ToolResult::text(format!(
                "sent sticker {} as message {id}",
                sticker.name
            )));
        }
        if let Some(file_id) = self.library.file_id(&name).await {
            let id = sender.send_sticker_id(&file_id).await?;
            return Ok(ToolResult::text(format!(
                "sent sticker {name} as message {id}"
            )));
        }
        Ok(ToolResult::error(format!(
            "no sticker named {name:?} — call list_stickers for valid names"
        )))
    }
}

/// Send an arbitrary file as media — photo, video, audio, voice, or document.
struct SendFile {
    router: Arc<ChatRouter>,
}

/// Arguments for `send_file`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendFileArgs {
    /// Path to a file on disk — media type follows its extension.
    path: Option<String>,
    /// A Telegram `file_id` from an inbound event, to resend media.
    file_id: Option<String>,
    /// With `file_id`: the media kind — `photo`, `video`, `audio`, `voice`,
    /// `animation`, `document`, or `sticker`. Defaults to `document`.
    kind: Option<String>,
    /// Optional caption (ignored for stickers).
    caption: Option<String>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for SendFile {
    type Arguments = SendFileArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "send_file".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Send a file to the chat: images go as photos (gif as animation), \
         videos as video, audio as audio (ogg as a voice note), everything \
         else as a document. Give `path` for a local file, or `file_id` plus \
         `kind` to resend media an event carried. `caption` is optional."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let caption = args.caption.as_deref();
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        if let Some(file_id) = args.file_id {
            let kind = parse_kind(args.kind.as_deref())?;
            let id = sender.send_file_id(kind, &file_id, caption).await?;
            return Ok(ToolResult::text(format!("sent media as message {id}")));
        }
        let Some(path) = args.path else {
            return Ok(ToolResult::error("give `path` or `file_id`"));
        };
        let id = sender
            .send_file(std::path::Path::new(&path), caption)
            .await?;
        Ok(ToolResult::text(format!("sent {path} as message {id}")))
    }
}

/// React to a message with an emoji (or clear the bot's reaction).
struct React {
    router: Arc<ChatRouter>,
}

/// Arguments for `react`.
#[derive(Debug, Deserialize, JsonSchema)]
struct ReactArgs {
    /// The message to react to.
    message_id: i64,
    /// The emoji (`👍`, `❤`, `🔥`, …). Omit to remove the bot's reaction.
    emoji: Option<String>,
    /// Play the big animation. Default false.
    is_big: Option<bool>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for React {
    type Arguments = ReactArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "react".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Set the bot's emoji reaction on a message — `emoji` like \"👍\"; \
         omit it to remove the reaction. `is_big` plays the large animation."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        sender
            .react(
                args.message_id,
                args.emoji.as_deref(),
                args.is_big.unwrap_or(false),
            )
            .await?;
        Ok(ToolResult::text(match args.emoji {
            Some(emoji) => format!("reacted {emoji} to message {}", args.message_id),
            None => format!("cleared reaction on message {}", args.message_id),
        }))
    }
}

/// Edit the text of a message the bot sent.
struct EditMessage {
    router: Arc<ChatRouter>,
}

/// Arguments for `edit_message`.
#[derive(Debug, Deserialize, JsonSchema)]
struct EditMessageArgs {
    /// The bot's message to edit (its id came back from send_message/reply).
    message_id: i64,
    /// The replacement text.
    text: String,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for EditMessage {
    type Arguments = EditMessageArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "edit_message".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Edit the text of a message the bot already sent — e.g. update a \
         progress note with the finished result."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        sender.edit(args.message_id, &args.text).await?;
        Ok(ToolResult::text(format!(
            "edited message {}",
            args.message_id
        )))
    }
}

/// Delete a message.
struct DeleteMessage {
    router: Arc<ChatRouter>,
}

/// Arguments for `delete_message`.
#[derive(Debug, Deserialize, JsonSchema)]
struct DeleteMessageArgs {
    /// The message to delete (the bot's own, or any where it can).
    message_id: i64,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for DeleteMessage {
    type Arguments = DeleteMessageArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "delete_message".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Delete a message — the bot's own anywhere, or others' where the \
         bot has delete rights."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        sender.delete_message(args.message_id).await?;
        Ok(ToolResult::text(format!(
            "deleted message {}",
            args.message_id
        )))
    }
}

/// Pin or unpin a message.
struct PinMessage {
    router: Arc<ChatRouter>,
}

/// Arguments for `pin_message`.
#[derive(Debug, Deserialize, JsonSchema)]
struct PinMessageArgs {
    /// The message to pin.
    message_id: i64,
    /// Remove the pin instead of setting it. Default false.
    unpin: Option<bool>,
    /// Notify the chat about the pin. Default true.
    notify: Option<bool>,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for PinMessage {
    type Arguments = PinMessageArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "pin_message".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Pin a message to the top of the chat (`unpin: true` removes the \
         pin; `notify: false` pins silently). Needs pin rights in groups."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        sender
            .pin(
                args.message_id,
                args.unpin.unwrap_or(false),
                args.notify.unwrap_or(true),
            )
            .await?;
        Ok(ToolResult::text(format!(
            "{} message {}",
            if args.unpin.unwrap_or(false) {
                "unpinned"
            } else {
                "pinned"
            },
            args.message_id
        )))
    }
}

/// Parse a `send_file` kind hint.
fn parse_kind(kind: Option<&str>) -> aither_core::Result<MediaKind> {
    Ok(match kind {
        None | Some("document") => MediaKind::Document,
        Some("photo") => MediaKind::Photo,
        Some("video") => MediaKind::Video,
        Some("audio") => MediaKind::Audio,
        Some("voice") => MediaKind::Voice,
        Some("animation") => MediaKind::Animation,
        Some("sticker") => MediaKind::Sticker,
        Some(other) => {
            return Err(aither_core::Error::msg(format!(
                "unknown media kind {other:?} — photo, video, audio, voice, \
                 animation, document, or sticker"
            )));
        }
    })
}

/// List the available stickers.
struct ListStickers {
    sticker_dir: PathBuf,
    library: Arc<StickerLibrary>,
}

/// Arguments for `list_stickers` (none).
#[derive(Debug, Deserialize, JsonSchema)]
struct ListStickersArgs {}

impl Tool for ListStickers {
    type Arguments = ListStickersArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "list_stickers".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "List every sendable sticker name: the local pack (writable — drop \
         an image file in and add a `name = \"meaning\"` line to its \
         `stickers.toml`, sendable immediately) plus library stickers \
         imported from Telegram sets (entries carrying a `set` field)."
            .into()
    }

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let mut entries = load_pack(&self.sticker_dir)?.list_json();
        let library: Vec<serde_json::Value> = self
            .library
            .list()
            .await
            .into_iter()
            .map(|(name, format, emoji, set)| {
                let mut entry = serde_json::json!({
                    "name": name,
                    "format": format,
                    "emoji": emoji,
                });
                if let Some(set) = set {
                    entry["set"] = serde_json::Value::String(set);
                }
                entry
            })
            .collect();
        if !library.is_empty() {
            let mut all: Vec<serde_json::Value> =
                serde_json::from_str(&entries).expect("pack list is an array");
            all.extend(library);
            entries = serde_json::to_string(&all).expect("sticker list serializes");
        }
        Ok(ToolResult::text(entries))
    }
}

/// Import every sticker of a Telegram set into the library.
struct ImportStickerSet {
    library: Arc<StickerLibrary>,
}

/// Arguments for `import_sticker_set`.
#[derive(Debug, Deserialize, JsonSchema)]
struct ImportStickerSetArgs {
    /// The set's short name or a `t.me/addstickers/<name>` link — visible
    /// on `sticker.set_name` when a sticker from the set arrives, or in the
    /// URL a sticker pack share produces.
    set: String,
}

impl Tool for ImportStickerSet {
    type Arguments = ImportStickerSetArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "import_sticker_set".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Import a whole Telegram sticker set into the library so every \
         sticker in it becomes sendable by name — the way to adopt a pack \
         someone else made. `set` is the short name (`sticker.set_name` on \
         an incoming sticker) or a `t.me/addstickers/<name>` link. Stickers \
         land as `<set>:<emoji>` names; the catalog persists across \
         restarts."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        match self.library.import_set(&args.set).await {
            Ok((title, names)) => Ok(ToolResult::text(format!(
                "imported {} stickers from {title:?} — sendable as {}",
                names.len(),
                names.join(", ")
            ))),
            Err(e) => Ok(ToolResult::error(e.to_string())),
        }
    }
}

/// Save one sticker an event carried under a chosen name.
struct SaveSticker {
    library: Arc<StickerLibrary>,
}

/// Arguments for `save_sticker`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SaveStickerArgs {
    /// The `file_id` the incoming `sticker` event carried.
    file_id: String,
    /// The name `send_sticker` will resolve it by — pick something
    /// memorable.
    name: String,
    /// The emoji it represents, when known (the event's `sticker.emoji`).
    emoji: Option<String>,
    /// The set it came from (the event's `sticker.set_name`), when known.
    set_name: Option<String>,
}

impl Tool for SaveSticker {
    type Arguments = SaveStickerArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "save_sticker".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Save a sticker an event carried into the library under a name you \
         choose — for keeping a single sticker you liked without importing \
         its whole set. `send_sticker` resolves the name from then on, \
         across restarts."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sticker = LibrarySticker {
            file_id: args.file_id,
            emoji: args.emoji,
            set_name: args.set_name,
            set_title: None,
            format: "static".to_string(),
        };
        match self.library.save_sticker(&args.name, sticker).await {
            Ok(()) => Ok(ToolResult::text(format!(
                "saved {} — sendable as `send_sticker` `{}`",
                args.name, args.name
            ))),
            Err(e) => Ok(ToolResult::error(e.to_string())),
        }
    }
}

/// List the Telegram sets imported into the library.
struct ListStickerSets {
    library: Arc<StickerLibrary>,
}

/// Arguments for `list_sticker_sets` (none).
#[derive(Debug, Deserialize, JsonSchema)]
struct ListStickerSetsArgs {}

impl Tool for ListStickerSets {
    type Arguments = ListStickerSetsArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "list_sticker_sets".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "List the Telegram sticker sets imported into the library: name, \
         title, and how many stickers each contributed."
            .into()
    }

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sets = self.library.list_sets().await;
        let entries: Vec<serde_json::Value> = sets
            .into_iter()
            .map(|(name, title, count)| {
                serde_json::json!({"name": name, "title": title, "stickers": count})
            })
            .collect();
        Ok(ToolResult::text(
            serde_json::to_string(&entries).expect("set list serializes"),
        ))
    }
}

/// Ask the daemon to reincarnate the shared agent process.
struct Restart {
    restart: ChanSender<()>,
}

/// Arguments for `restart` (none).
#[derive(Debug, Deserialize, JsonSchema)]
struct RestartArgs {}

impl Tool for Restart {
    type Arguments = RestartArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "restart".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Restart your agent process. Use after editing AGENTS.md or adding \
         skills so the new instructions load — your ACP session and all files \
         persist, so the conversation continues seamlessly. The restart \
         happens once this turn ends; finish any messages you owe the chat \
         first."
            .into()
    }

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<Self::Res> {
        self.restart
            .try_send(())
            .map_err(|e| aither_core::Error::msg(format!("restart channel rejected: {e}")))?;
        Ok(ToolResult::text(
            "restart scheduled — end your turn normally",
        ))
    }
}

/// Time bounds shared by `history` and `search_history`.
#[derive(Debug, Deserialize, JsonSchema)]
struct TimeRangeArgs {
    /// Records at/after this time: epoch seconds, RFC3339
    /// (`2026-09-12T10:00:00Z`), or relative back from now (`30m`, `2h`,
    /// `7d`). Omit for no lower bound.
    since: Option<String>,
    /// Records before this time, same formats. Omit for "now".
    until: Option<String>,
    /// At most this many records, newest kept (default 50).
    limit: Option<usize>,
}

impl TimeRangeArgs {
    /// Resolve the bounds to epoch seconds, defaulting `limit` to 50.
    fn bounds(&self) -> Result<(Option<i64>, Option<i64>, usize), String> {
        let now = crate::sender::epoch_secs();
        let parse = |arg: &Option<String>| -> Result<Option<i64>, String> {
            arg.as_deref()
                .map(|s| crate::history::parse_time_arg(s, now))
                .transpose()
        };
        Ok((
            parse(&self.since)?,
            parse(&self.until)?,
            self.limit.unwrap_or(50),
        ))
    }
}

/// Render records as a JSON array the agent reads directly — `ts` also
/// formatted as ISO time so it never has to convert epochs by hand.
fn render_records(records: Vec<serde_json::Value>) -> ToolResult {
    let rendered: Vec<serde_json::Value> = records
        .into_iter()
        .map(|mut r| {
            if let Some(ts) = r["ts"].as_i64()
                && let Ok(t) = jiff::Timestamp::from_second(ts)
            {
                r["time"] = serde_json::Value::from(t.to_string());
            }
            r
        })
        .collect();
    ToolResult::text(serde_json::to_string(&rendered).expect("records serialize"))
}

/// Ask the platform whether a known message still exists — the only way
/// to learn a message was deleted, since Telegram reports no deletion
/// event to bots.
struct MessageStatus {
    router: Arc<ChatRouter>,
}

/// Arguments for `message_status`.
#[derive(Debug, Deserialize, JsonSchema)]
struct MessageStatusArgs {
    /// The message id to check — from an event or a history record.
    message_id: i64,
    /// Target chat — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for MessageStatus {
    type Arguments = MessageStatusArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "message_status".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Check whether a message still exists in the chat. Deletions are never \
         pushed to you — if you need to know whether a message (yours or a \
         user's) is still there, ask. Returns `{\"exists\": bool}`."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let sender = match target(&self.router, args.chat.as_deref()) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        let exists = sender.probe_message(args.message_id).await?;
        Ok(ToolResult::text(
            serde_json::json!({"message_id": args.message_id, "exists": exists}).to_string(),
        ))
    }
}

/// The most outbound records an IM-record read probes for deletion —
/// keeps a `history` call from turning into an API storm.
const PROBE_CAP: usize = 8;

/// Mark records `"deleted": true` once their message is known gone.
/// Lazily probes the newest unprobed `message_id`s (up to `PROBE_CAP`);
/// dead ids are cached on `history` and never probed again.
async fn annotate_deleted(records: &mut [serde_json::Value], sender: &Sender, history: &History) {
    let unprobed: Vec<i64> = records
        .iter()
        .rev()
        .filter_map(|r| r["message_id"].as_i64())
        .filter(|id| !history.is_deleted(*id))
        .collect();
    for id in unprobed.into_iter().take(PROBE_CAP) {
        if let Ok(false) = sender.probe_message(id).await {
            history.mark_deleted(id);
        }
    }
    for r in records.iter_mut() {
        if r["message_id"]
            .as_i64()
            .is_some_and(|id| history.is_deleted(id))
        {
            r["deleted"] = serde_json::Value::from(true);
        }
    }
}

/// Look up a chat's metadata and the bot's access to it.
struct ChatInfo {
    router: Arc<ChatRouter>,
}

/// Arguments for `chat_info`.
#[derive(Debug, Deserialize, JsonSchema)]
struct ChatInfoArgs {
    /// The chat to look up: a numeric id, an @username, or a t.me link
    /// (`t.me/<name>`, `t.me/c/<id>` — a message link works too).
    chat: String,
}

impl Tool for ChatInfo {
    type Arguments = ChatInfoArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "chat_info".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Look up a chat through the bot's Telegram identity: type, title, \
         description, member count, and `bot_status` — whether the bot sits \
         in it. `chat` takes a numeric id, an @username, or a t.me link \
         (invite links cannot resolve). `readable: true` means the bot is a \
         member, so its events reach you and `history`/`fetch_message` work \
         there; for a chat you can't reach, ask the user to add the bot."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        // Any sender is a valid platform handle — the lookup isn't bound to
        // the chat it was invoked from; the current chat's sender is used.
        let sender = match target(&self.router, None) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        match sender.chat_info(&args.chat).await {
            Ok(info) => Ok(ToolResult::text(info.to_string())),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

/// Read one message out of a chat by id or link.
struct FetchMessage {
    router: Arc<ChatRouter>,
}

/// Arguments for `fetch_message`.
#[derive(Debug, Deserialize, JsonSchema)]
struct FetchMessageArgs {
    /// The chat holding the message: a numeric id, an @username, or a t.me
    /// link — a message link (`t.me/<name>/<id>`, `t.me/c/<id>/<msg>`)
    /// supplies `message_id` itself.
    chat: String,
    /// The message id — required unless `chat` is a message link.
    message_id: Option<i64>,
}

impl Tool for FetchMessage {
    type Arguments = FetchMessageArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "fetch_message".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Read one message from a chat the bot belongs to: briefly forwards it \
         into the current chat to read it, then deletes the copy. `chat` is \
         the SOURCE — numeric id, @username, or a t.me message link (which \
         carries the message id). Only membership chats answer: for one the \
         bot isn't in, the call fails — ask the user to add the bot instead."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        // The scratch copy lands in the current chat — `chat` names the
        // source, not the routing target.
        let sender = match target(&self.router, None) {
            Ok(sender) => sender,
            Err(result) => return Ok(result),
        };
        match sender.fetch_message(&args.chat, args.message_id).await {
            Ok(message) => Ok(ToolResult::text(message.to_string())),
            Err(error) => Ok(ToolResult::error(error.to_string())),
        }
    }
}

/// Read back a chat's IM record: every inbound event and outbound action,
/// newest `limit` inside the range.
struct ChatHistory {
    router: Arc<ChatRouter>,
}

/// Arguments for `history`.
#[derive(Debug, Deserialize, JsonSchema)]
struct HistoryArgs {
    #[serde(flatten)]
    range: TimeRangeArgs,
    /// Whose record — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for ChatHistory {
    type Arguments = HistoryArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "history".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Pull a chat's IM record: every message the bot saw arrive and \
         everything it sent there, as JSON records with `ts`/`time`, `dir` \
         (in/out), `from`, `text`. This is the durable log of the \
         conversation — it predates and outlives your session. \
         `since`/`until` accept epoch seconds, RFC3339, or relative \
         `30m`/`2h`/`7d`; `limit` (default 50) keeps the newest. `chat` \
         selects which chat's record — omit for the chat the current \
         events came from. Use it to recall what happened before a restart \
         wiped your context, or to answer \"what did we say about X \
         yesterday\"."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let (since, until, limit) = match args.range.bounds() {
            Ok(bounds) => bounds,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let (history, sender) = match self.router.history_for(args.chat.as_deref()) {
            Ok(target) => target,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let mut records = history.tail(since, until, limit);
        annotate_deleted(&mut records, &sender, &history).await;
        Ok(render_records(records))
    }
}

/// Search a chat's IM record for text.
struct SearchHistory {
    router: Arc<ChatRouter>,
}

/// Arguments for `search_history`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SearchHistoryArgs {
    /// Case-insensitive substring matched against `text`, `from.name`,
    /// and `command` fields.
    query: String,
    #[serde(flatten)]
    range: TimeRangeArgs,
    /// Whose record — the `chat` id from an event, or `platform:id`.
    /// Omit for the chat the current events came from.
    chat: Option<String>,
}

impl Tool for SearchHistory {
    type Arguments = SearchHistoryArgs;
    type Res = ToolResult;

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "search_history".into()
    }

    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Search a chat's IM record — case-insensitive match on message text, \
         sender name, and command fields, over the durable log of what was \
         actually said. Same `since`/`until`/`limit` as `history`; `chat` \
         selects which chat (default: the one the current events came \
         from). The fastest way to answer \"when did X mention Y\" or \
         \"did I already reply to that\"."
            .into()
    }

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let (since, until, limit) = match args.range.bounds() {
            Ok(bounds) => bounds,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let (history, sender) = match self.router.history_for(args.chat.as_deref()) {
            Ok(target) => target,
            Err(msg) => return Ok(ToolResult::error(msg)),
        };
        let mut records = history.search(&args.query, since, until, limit);
        annotate_deleted(&mut records, &sender, &history).await;
        Ok(render_records(records))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tests::test_router;
    use crate::chat::ChatKey;
    use async_channel::Receiver;

    fn tempdir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("acpbot-tools-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_key() -> ChatKey {
        ChatKey {
            platform: "telegram",
            id: "0".to_string(),
        }
    }

    /// The parts a test may care about: the toolset, the router behind it,
    /// the senders' recorded calls, the chat keys senders were bound to,
    /// the restart signal, and the test chat's IM record.
    type Fixture = (
        Tools,
        Arc<ChatRouter>,
        Receiver<String>,
        Receiver<String>,
        Receiver<()>,
        Arc<History>,
    );

    /// A toolset served through a router whose senders record to `out`;
    /// `keys` reports which chat each lazily-built sender was bound to.
    /// The returned receivers must outlive the calls or the channels
    /// close underneath them.
    fn fixture(dir: &std::path::Path) -> Fixture {
        let (out_tx, out_rx) = async_channel::unbounded();
        let (restart_tx, restart_rx) = async_channel::unbounded();
        let (router, keys) = test_router(dir, out_tx);
        router.set_current(test_key());
        let history = router.history(&test_key());
        (
            chat_tools(router.clone(), restart_tx),
            router,
            out_rx,
            keys,
            restart_rx,
            history,
        )
    }

    /// A sticker dropped into the pack mid-session — a file plus a
    /// `stickers.toml` meaning — is listable and sendable immediately.
    #[test]
    fn sticker_pack_hot_reloads() {
        let dir = tempdir("hot-reload");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);
        let pack_dir = dir.join("stickers");
        std::fs::create_dir_all(&pack_dir).unwrap();

        let listed = futures_lite::future::block_on(tools.call("list_stickers", "{}")).unwrap();
        assert_eq!(listed.as_text(), Some("[]"));

        std::fs::write(pack_dir.join("smug.png"), b"png-bytes").unwrap();
        std::fs::write(pack_dir.join("stickers.toml"), "smug = \"a smug face\"\n").unwrap();

        let listed = futures_lite::future::block_on(tools.call("list_stickers", "{}")).unwrap();
        let text = listed.as_text().unwrap();
        assert!(text.contains("smug"), "new sticker not listed: {text}");

        let sent =
            futures_lite::future::block_on(tools.call("send_sticker", "{\"name\":\"smug\"}"))
                .unwrap();
        assert!(
            sent.as_text().unwrap().contains("sent sticker"),
            "unexpected result: {}",
            sent.as_text().unwrap_or("<none>")
        );
        let outbound = out.try_recv().unwrap();
        assert_eq!(outbound, "sticker:smug");
    }

    /// A sticker saved into the library is sendable by its chosen name and
    /// shows up in `list_stickers` — with no Telegram client needed.
    #[test]
    fn saved_sticker_sends_by_name() {
        let dir = tempdir("save-sticker");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);

        let saved = futures_lite::future::block_on(tools.call(
            "save_sticker",
            "{\"file_id\":\"CAAC_save\",\"name\":\"my_laugh\",\"emoji\":\"😂\"}",
        ))
        .unwrap();
        assert!(
            saved.as_text().unwrap().contains("saved my_laugh"),
            "unexpected: {}",
            saved.as_text().unwrap_or("<none>")
        );

        // The name now resolves through the library.
        let sent =
            futures_lite::future::block_on(tools.call("send_sticker", "{\"name\":\"my_laugh\"}"))
                .unwrap();
        assert!(sent.as_text().unwrap().contains("sent sticker"));
        assert_eq!(out.try_recv().unwrap(), "sticker_id:CAAC_save");

        // … and the library lists it.
        let listed = futures_lite::future::block_on(tools.call("list_stickers", "{}")).unwrap();
        assert!(listed.as_text().unwrap().contains("my_laugh"));

        // Duplicate names are refused, not clobbered.
        let dup = futures_lite::future::block_on(
            tools.call("save_sticker", "{\"file_id\":\"X\",\"name\":\"my_laugh\"}"),
        )
        .unwrap();
        assert!(dup.as_text().unwrap().contains("already taken"));
    }

    /// The MCP surface the agent sees — guard against a registration being
    /// dropped in a refactor.
    #[test]
    fn registered_tool_names() {
        let dir = tempdir("tool-names");
        let (tools, _router, _out, _keys, _restart, _history) = fixture(&dir);
        let mut names: Vec<String> = tools
            .definitions()
            .iter()
            .map(|d| d.name().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "chat_info",
                "delete_message",
                "edit_message",
                "fetch_message",
                "history",
                "import_sticker_set",
                "list_sticker_sets",
                "list_stickers",
                "message_status",
                "pin_message",
                "react",
                "reply",
                "restart",
                "save_sticker",
                "search_history",
                "send_file",
                "send_message",
                "send_sticker",
            ]
        );
    }

    /// `history`/`search_history` read the IM record the sender writes —
    /// a send lands in `history.jsonl` as an `out` record immediately.
    #[test]
    fn history_reads_back_outbound() {
        let dir = tempdir("history");
        let (tools, _router, _out, _keys, _restart, _history) = fixture(&dir);
        let block = |name, args| futures_lite::future::block_on(tools.call(name, args)).unwrap();

        block("send_message", "{\"text\":\"morning all\"}");
        let text = block("history", "{}").as_text().unwrap().to_string();
        assert!(text.contains("\"dir\":\"out\""), "{text}");
        assert!(text.contains("morning all"), "{text}");
        assert!(text.contains("\"time\":\"20"), "{text}");

        let text = block("search_history", "{\"query\":\"MORNING\"}")
            .as_text()
            .unwrap()
            .to_string();
        assert!(text.contains("morning all"), "{text}");
        let text = block("search_history", "{\"query\":\"zzz\"}")
            .as_text()
            .unwrap()
            .to_string();
        assert_eq!(text, "[]");

        // A bad time argument is a usage error, not a panic.
        let result = block("history", "{\"since\":\"yesterdayish\"}");
        assert!(result.is_error());
    }

    /// `message_status` probes existence; `history` marks records whose
    /// message is known deleted — never an inbound event.
    #[test]
    fn message_status_and_deleted_annotation() {
        let dir = tempdir("status");
        let (tools, _router, _out, _keys, _restart, history) = fixture(&dir);
        let block = |name, args| futures_lite::future::block_on(tools.call(name, args)).unwrap();

        // The Record platform reports every message alive.
        let text = block("message_status", "{\"message_id\":42}")
            .as_text()
            .unwrap()
            .to_string();
        assert!(text.contains("\"exists\":true"), "{text}");

        // A sent message (Record reports id 0) found deleted gets marked.
        block("send_message", "{\"text\":\"will vanish\"}");
        history.mark_deleted(0);
        let text = block("history", "{}").as_text().unwrap().to_string();
        assert!(text.contains("\"deleted\":true"), "{text}");
    }

    /// `send_sticker` by `file_id` resends a sticker Telegram already hosts.
    #[test]
    fn send_sticker_by_file_id() {
        let dir = tempdir("sticker-id");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);
        let result = futures_lite::future::block_on(
            tools.call("send_sticker", "{\"file_id\":\"CAACAgEAAxk\"}"),
        )
        .unwrap();
        assert!(result.as_text().unwrap().contains("sent sticker"));
        assert_eq!(out.try_recv().unwrap(), "sticker_id:CAACAgEAAxk");
    }

    /// `send_file` sends a local path and resends a hosted `file_id`.
    #[test]
    fn send_file_by_path_and_id() {
        let dir = tempdir("send-file");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);

        let result = futures_lite::future::block_on(tools.call(
            "send_file",
            "{\"path\":\"/tmp/report.pdf\",\"caption\":\"here\"}",
        ))
        .unwrap();
        assert!(result.as_text().unwrap().contains("sent /tmp/report.pdf"));
        assert_eq!(out.try_recv().unwrap(), "file:/tmp/report.pdf");

        let result = futures_lite::future::block_on(
            tools.call("send_file", "{\"file_id\":\"BQADBQAD\",\"kind\":\"video\"}"),
        )
        .unwrap();
        assert!(result.as_text().unwrap().contains("sent media"));
        assert_eq!(out.try_recv().unwrap(), "file_id:Video:BQADBQAD");

        // Neither `path` nor `file_id` is a usage error, not a panic.
        let result = futures_lite::future::block_on(tools.call("send_file", "{}")).unwrap();
        assert!(result.is_error());
    }

    /// The message-management tools hit the platform calls they wrap.
    #[test]
    fn message_management_tools() {
        let dir = tempdir("manage");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);
        let block = |args| futures_lite::future::block_on(tools.call("react", args)).unwrap();

        let result = block("{\"message_id\":9,\"emoji\":\"👍\"}");
        assert!(result.as_text().unwrap().contains("reacted 👍"));
        assert_eq!(out.try_recv().unwrap(), "react:9:👍");

        let result = block("{\"message_id\":9}");
        assert!(result.as_text().unwrap().contains("cleared"));
        assert_eq!(out.try_recv().unwrap(), "react:9:");

        let result = futures_lite::future::block_on(
            tools.call("edit_message", "{\"message_id\":9,\"text\":\"done\"}"),
        )
        .unwrap();
        assert!(result.as_text().unwrap().contains("edited"));
        assert_eq!(out.try_recv().unwrap(), "edit:9:done");

        let result =
            futures_lite::future::block_on(tools.call("delete_message", "{\"message_id\":9}"))
                .unwrap();
        assert!(result.as_text().unwrap().contains("deleted"));
        assert_eq!(out.try_recv().unwrap(), "delete:9");

        let result = futures_lite::future::block_on(
            tools.call("pin_message", "{\"message_id\":9,\"notify\":false}"),
        )
        .unwrap();
        assert!(result.as_text().unwrap().contains("pinned"));
        assert_eq!(out.try_recv().unwrap(), "pin:9:false");

        let result = futures_lite::future::block_on(
            tools.call("pin_message", "{\"message_id\":9,\"unpin\":true}"),
        )
        .unwrap();
        assert!(result.as_text().unwrap().contains("unpinned"));
        assert_eq!(out.try_recv().unwrap(), "pin:9:true");
    }

    /// `restart` schedules a reincarnation — a unit lands on the channel
    /// and the tool tells the agent to wrap up its turn.
    #[test]
    fn restart_signals_channel() {
        let dir = tempdir("restart");
        let (tools, _router, _out, _keys, restart_rx, _history) = fixture(&dir);
        let result = futures_lite::future::block_on(tools.call("restart", "{}")).unwrap();
        assert_eq!(
            result.as_text(),
            Some("restart scheduled — end your turn normally")
        );
        assert!(restart_rx.try_recv().is_ok());
    }

    /// `buttons` rows validate through the tool: each button takes exactly
    /// one of `data` or `url` — anything else is a usage error, not a panic.
    #[test]
    fn send_message_buttons_validate() {
        let dir = tempdir("buttons");
        let (tools, _router, out, _keys, _restart, _history) = fixture(&dir);

        let result = futures_lite::future::block_on(tools.call(
            "send_message",
            r#"{"text":"pick","buttons":[[{"text":"yes","data":"y"},{"text":"no","data":"n"}],[{"text":"link","url":"https://example.com"}]]}"#,
        ))
        .unwrap();
        assert!(!result.is_error());
        assert_eq!(out.try_recv().unwrap(), "send:pick");

        // Both `data` and `url` on one button is rejected…
        let result = futures_lite::future::block_on(tools.call(
            "send_message",
            r#"{"text":"x","buttons":[[{"text":"bad","data":"d","url":"https://example.com"}]]}"#,
        ))
        .unwrap();
        assert!(result.is_error());

        // …and so is neither.
        let result = futures_lite::future::block_on(tools.call(
            "send_message",
            r#"{"text":"x","buttons":[[{"text":"bad"}]]}"#,
        ))
        .unwrap();
        assert!(result.is_error());
    }

    /// The `chat` argument picks the conversation a call lands in: absent
    /// it is the turn's current chat, present it is that chat — and a bad
    /// target is a usage error, not a send to the wrong room.
    #[test]
    fn chat_arg_routes() {
        let dir = tempdir("route");
        let (tools, router, out, keys, _restart, _history) = fixture(&dir);
        let block = |name, args| futures_lite::future::block_on(tools.call(name, args)).unwrap();

        // Default: the turn's current chat ("0") gets the sender.
        let result = block("send_message", "{\"text\":\"home\"}");
        assert!(!result.is_error());
        assert_eq!(out.try_recv().unwrap(), "send:home");
        assert_eq!(keys.try_recv().unwrap(), "0");

        // Explicit `chat` builds and targets that chat's sender…
        let result = block("send_message", "{\"text\":\"away\",\"chat\":\"7\"}");
        assert!(!result.is_error());
        assert_eq!(out.try_recv().unwrap(), "send:away");
        assert_eq!(keys.try_recv().unwrap(), "7");

        // …and a `platform:id` form resolves to the same cached sender.
        let result = block(
            "send_message",
            "{\"text\":\"again\",\"chat\":\"telegram:7\"}",
        );
        assert!(!result.is_error());
        assert_eq!(out.try_recv().unwrap(), "send:again");
        assert!(keys.try_recv().is_err(), "sender rebuilt for cached chat");

        // A chat the daemon doesn't serve can never be addressed.
        let result = block("send_message", "{\"text\":\"x\",\"chat\":\"discord:9\"}");
        assert!(result.is_error());

        // `history` follows `chat` too: send into "7", read its record.
        let text = block("history", "{\"chat\":\"7\"}")
            .as_text()
            .unwrap()
            .to_string();
        assert!(text.contains("away"), "{text}");
        let text = block("history", "{}").as_text().unwrap().to_string();
        assert!(!text.contains("away"), "{text}");
        drop(router);
    }
}
