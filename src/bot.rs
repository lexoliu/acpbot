//! Telegram adapter wiring: every update becomes a [`ChatEvent`] on the
//! dispatcher's channel.
//!
//! Handlers must return fast — Telegram's dispatcher processes updates one at
//! a time — so they only build the event and push it into the channel; the
//! shared actor performs the actual ACP turn.

use async_channel::Sender;
use botkit_cli::{CliBot, CliContextData, Transport as CliTransport};
use botkit_core::{Context, Response};
use botkit_discord::{
    DiscordBot, DiscordContextData, GatewayIntents, InteractionData, MessageContextData,
};
use botkit_telegram::types::{ChatType, EntityType, Message, Update, UpdateKind};
use botkit_telegram::{TelegramBot, TelegramContextData};
use tracing::warn;

use crate::chat::{
    ChatEvent, EventCommand, EventMedia, EventReaction, EventReplyRef, EventSender, EventSticker,
};

/// The bot's own Telegram identity (from `getMe`): group events are
/// classified against it — a reply to `id` or a mention of `username`
/// means the message is `direct`, everything else in a group is `ambient`.
#[derive(Debug, Clone)]
pub struct BotIdentity {
    /// Telegram user id of the bot account.
    pub id: i64,
    /// The bot's username without `@`, when it has one.
    pub username: Option<String>,
}

/// The bot's own Discord identity (from `GET /users/@me`): guild messages
/// are classified against it — a reply to the bot's message or an
/// `@<id>` mention is `direct`, everything else in a guild is `ambient`.
/// DMs are always direct.
#[derive(Debug, Clone)]
pub struct DiscordIdentity {
    /// Snowflake user id of the bot account.
    pub id: String,
}

/// Which platform's identity group events are classified against.
#[derive(Debug, Clone)]
enum Identity {
    /// The CLI platform — attention arrives on the wire, nothing to
    /// classify against.
    None,
    Telegram(BotIdentity),
    Discord(DiscordIdentity),
}

/// A [`TelegramBot`] that forwards every event to `events`.
pub fn build(token: String, events: Sender<ChatEvent>, me: BotIdentity) -> TelegramBot {
    let forward = {
        let events = events.clone();
        move |ctx: Context| {
            let events = events.clone();
            let me = Identity::Telegram(me.clone());
            async move {
                if let Err(error) = forward(ctx, &events, me).await {
                    warn!(%error, "failed to forward chat event");
                }
                Response::empty()
            }
        }
    };

    TelegramBot::new(token)
        // Nothing is registered with Telegram: commands and buttons are data
        // for the agent, not a menu.
        .skip_command_registration()
        .message(forward.clone())
        .button("*", forward.clone())
        .fallback(forward)
}

/// A bot running on the agent-facing CLI backend.
///
/// `transport` selects how an external driver injects events and reads
/// actions (stdio lines or a unix socket); everything else behaves like the
/// telegram path — same handlers, same `ChatEvent` extraction, same
/// dispatcher.
pub fn build_cli(transport: CliTransport, events: Sender<ChatEvent>) -> CliBot {
    let forward = {
        let events = events.clone();
        move |ctx: Context| {
            let events = events.clone();
            async move {
                if let Err(error) = forward(ctx, &events, Identity::None).await {
                    warn!(%error, "failed to forward chat event");
                }
                Response::empty()
            }
        }
    };

    CliBot::new(transport)
        .message(forward.clone())
        .button("*", forward.clone())
        .fallback(forward)
}

/// A [`DiscordBot`] that forwards every event to `events`.
///
/// Slash commands and buttons reach the forwarder through `fallback` —
/// nothing is registered with Discord; commands and buttons are data for
/// the agent, not a menu.
pub fn build_discord(
    token: String,
    application_id: String,
    events: Sender<ChatEvent>,
    me: DiscordIdentity,
) -> DiscordBot {
    let forward = {
        let events = events.clone();
        move |ctx: Context| {
            let events = events.clone();
            let me = Identity::Discord(me.clone());
            async move {
                if let Err(error) = forward(ctx, &events, me).await {
                    warn!(%error, "failed to forward chat event");
                }
                Response::empty()
            }
        }
    };

    DiscordBot::new(
        token,
        application_id,
        // Guild + DM messages, their contents, and reactions on both.
        GatewayIntents::GUILDS
            | GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT
            | GatewayIntents::GUILD_MESSAGE_REACTIONS
            | GatewayIntents::DIRECT_MESSAGE_REACTIONS,
    )
    .skip_command_registration()
    .message(forward.clone())
    .button("*", forward.clone())
    .fallback(forward)
}

/// Build the [`ChatEvent`] a `Context` describes and enqueue it.
async fn forward(
    ctx: Context,
    events: &Sender<ChatEvent>,
    me: Identity,
) -> Result<(), EventsClosed> {
    events
        .send(to_event(&ctx, &me))
        .await
        .map_err(|_| EventsClosed)
}

/// The dispatcher's event channel closed — the daemon is shutting down.
#[derive(Debug)]
struct EventsClosed;

impl std::fmt::Display for EventsClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("event channel closed")
    }
}

impl std::error::Error for EventsClosed {}

/// Translate the unified context plus the platform-specific payload into the
/// JSON event the agent receives.
fn to_event(ctx: &Context, me: &Identity) -> ChatEvent {
    let platform: &str = if ctx.platform::<CliContextData>().is_some() {
        "cli"
    } else if ctx.platform::<MessageContextData>().is_some()
        || ctx.platform::<DiscordContextData>().is_some()
    {
        "discord"
    } else {
        "telegram"
    };
    let base = ChatEvent {
        kind: "message".into(),
        platform: platform.into(),
        chat: ctx.channel_id().to_string(),
        ts: crate::sender::epoch_secs(),
        attention: "direct".into(),
        chat_type: None,
        chat_title: None,
        message_id: None,
        from: EventSender {
            id: ctx.user_id().to_string(),
            name: ctx.user_name().to_string(),
        },
        text: ctx.message_content().map(str::to_string),
        command: None,
        button: None,
        reply_to: None,
        sticker: None,
        media: None,
        reaction: None,
        thread_id: None,
    };
    if let Some(data) = ctx.platform::<CliContextData>() {
        return extract_cli(data, base);
    }
    if let Some(data) = ctx.platform::<MessageContextData>() {
        return extract_discord_message(data, base, me);
    }
    if let Some(data) = ctx.platform::<DiscordContextData>() {
        return extract_discord_interaction(data, base);
    }
    match ctx.platform::<TelegramContextData>() {
        Some(data) => {
            let Identity::Telegram(me) = me else {
                unreachable!("telegram build always supplies the bot identity")
            };
            extract_telegram(ctx, data, base, me)
        }
        None => extract_generic(ctx, base),
    }
}

/// The CLI backend carries its whole wire payload in the context: real file
/// paths (whose bytes `fetch_media` copies into `inbox/`), thread ids,
/// stickers, and reaction diffs.
fn extract_cli(data: &CliContextData, mut event: ChatEvent) -> ChatEvent {
    use botkit_cli::Inbound;
    match &data.event {
        Inbound::Message(message) => {
            event.kind = "message".into();
            event.message_id = message.message_id;
            event.text = message.text.clone().or_else(|| message.caption.clone());
            event.thread_id = message.thread_id;
            if message.ambient {
                event.attention = "ambient".into();
            }
            event.reply_to = message.reply_to.as_ref().map(|replied| EventReplyRef {
                message_id: replied.message_id,
                from: replied.from.clone(),
                text: replied.text.clone(),
                // The CLI wire ref carries no attachment detail.
                sticker: None,
                media: None,
            });
            event.media = message.files.first().map(|file| EventMedia {
                kind: match file.kind.as_str() {
                    "photo" => "photo".into(),
                    "video" => "video".into(),
                    "audio" => "audio".into(),
                    "voice" => "voice".into(),
                    "animation" => "animation".into(),
                    _ => "document".into(),
                },
                // The wire file's resend id defaults to its local path —
                // `fetch_media` resolves either to bytes under `inbox/`.
                file_id: file.file_id.clone().unwrap_or_else(|| file.path.clone()),
                file: None,
            });
            event.sticker = message.sticker.as_ref().map(|sticker| EventSticker {
                file_id: sticker.file_id.clone(),
                emoji: sticker.emoji.clone(),
                set_name: sticker.set_name.clone(),
                format: match sticker.format.as_str() {
                    "animated" => "animated".into(),
                    "video" => "video".into(),
                    _ => "static".into(),
                },
                file: None,
            });
        }
        Inbound::Command(command) => {
            event.kind = "command".into();
            event.message_id = command.message_id;
            event.thread_id = command.thread_id;
            event.command = Some(EventCommand {
                name: command.name.clone(),
                args: if command.args.is_empty() {
                    None
                } else {
                    Some(command.args.clone())
                },
            });
        }
        Inbound::Button(button) => {
            event.kind = "button".into();
            event.message_id = button.message_id;
            event.text = button.message_text.clone();
            event.button = Some(button.data.clone());
            event.thread_id = button.thread_id;
        }
        Inbound::Reaction(reaction) => {
            event.kind = "reaction".into();
            event.message_id = Some(reaction.message_id);
            event.reaction = Some(EventReaction {
                added: reaction.added.clone(),
                removed: reaction.removed.clone(),
            });
        }
        Inbound::Edited(edited) => {
            event.kind = "edited".into();
            event.message_id = Some(edited.message_id);
            event.text = edited.text.clone();
            event.thread_id = edited.thread_id;
        }
        Inbound::Subscribe => {}
    }
    event
}

/// A Discord `Message` (new or edited) becomes a message event: guild
/// channels map to `group`, DMs to `private`, and snowflake ids to the
/// `i64` the schema carries.
fn extract_discord_message(
    data: &MessageContextData,
    mut event: ChatEvent,
    me: &Identity,
) -> ChatEvent {
    let message = data.message();
    event.kind = if data.edited { "edited" } else { "message" }.into();
    event.message_id = Some(snowflake(&message.id));
    if let Some(ts) = discord_ts(message) {
        event.ts = ts;
    }
    event.text = (!message.content.is_empty()).then(|| message.content.clone());
    event.chat_type = Some(
        if message.guild_id.is_some() {
            "group"
        } else {
            "private"
        }
        .into(),
    );
    event.attention = discord_attention(message, me).into();
    event.reply_to = message.referenced_message.as_ref().map(|m| EventReplyRef {
        message_id: snowflake(&m.id),
        from: Some(
            m.author
                .global_name
                .clone()
                .unwrap_or_else(|| m.author.username.clone()),
        ),
        text: (!m.content.is_empty()).then(|| m.content.clone()),
        sticker: None,
        media: m.attachments.first().map(discord_event_media),
    });
    // `file_id` carries the attachment's CDN URL; `fetch_media` downloads it.
    event.media = message.attachments.first().map(discord_event_media);
    event
}

/// A Discord interaction — slash commands land in `command`, component
/// presses in `button` (`custom_id` as the button payload, the pressed
/// message's id and text alongside).
fn extract_discord_interaction(data: &DiscordContextData, mut event: ChatEvent) -> ChatEvent {
    // Interactions only exist on things aimed at the bot.
    event.attention = "direct".into();
    let interaction = data.interaction();
    event.chat_type = Some(
        if interaction.guild_id.is_some() {
            "group"
        } else {
            "private"
        }
        .into(),
    );
    match &interaction.data {
        Some(InteractionData::ApplicationCommand { name, options, .. }) => {
            event.kind = "command".into();
            event.command = Some(EventCommand {
                name: name.clone(),
                args: (!options.is_empty()).then(|| {
                    options
                        .iter()
                        .map(|o| match &o.value {
                            serde_json::Value::String(s) => format!("{}={s}", o.name),
                            other => format!("{}={other}", o.name),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                }),
            });
        }
        Some(
            InteractionData::MessageComponent { custom_id, .. }
            | InteractionData::ModalSubmit { custom_id, .. },
        ) => {
            event.kind = "button".into();
            event.button = Some(custom_id.clone());
            if let Some(message) = &interaction.message {
                event.message_id = Some(snowflake(&message.id));
                event.text = (!message.content.is_empty()).then(|| message.content.clone());
            }
        }
        None => {}
    }
    event
}

/// `"direct"` when a guild message addresses the bot — a reply to one of
/// its messages or an `@`-mention of its id — and on every DM; other
/// guild traffic is room context.
fn discord_attention(message: &botkit_discord::Message, me: &Identity) -> &'static str {
    let Identity::Discord(me) = me else {
        return "direct";
    };
    if message.guild_id.is_none() {
        return "direct";
    }
    let replied_to_bot = message
        .referenced_message
        .as_ref()
        .is_some_and(|m| m.author.id == me.id);
    let mentioned = message.mentions.iter().any(|u| u.id == me.id);
    if replied_to_bot || mentioned {
        "direct"
    } else {
        "ambient"
    }
}

/// A Discord attachment as `EventMedia` — `file_id` carries the CDN URL.
fn discord_event_media(attachment: &botkit_discord::Attachment) -> EventMedia {
    EventMedia {
        kind: discord_media_kind(attachment).into(),
        file_id: attachment.url.clone(),
        file: None,
    }
}

/// The event-schema media kind a Discord attachment's content type implies.
fn discord_media_kind(attachment: &botkit_discord::Attachment) -> &'static str {
    match attachment
        .content_type
        .as_deref()
        .and_then(|ct| ct.split('/').next())
    {
        Some("image") => "photo",
        Some("video") => "video",
        Some("audio") => "audio",
        _ => "document",
    }
}

/// A Discord snowflake as the `i64` the event schema carries — the same
/// conversion outbound sends use to name messages.
fn snowflake(id: &str) -> i64 {
    crate::sender::snowflake_id(id)
}

/// Epoch seconds from Discord's ISO 8601 timestamp — `edited_timestamp`
/// on edits, like Telegram's `edit_date`.
fn discord_ts(message: &botkit_discord::Message) -> Option<i64> {
    let stamp = message
        .edited_timestamp
        .as_deref()
        .unwrap_or(&message.timestamp);
    stamp.parse::<jiff::Timestamp>().ok().map(|t| t.as_second())
}

fn extract_telegram(
    ctx: &Context,
    data: &TelegramContextData,
    mut event: ChatEvent,
    me: &BotIdentity,
) -> ChatEvent {
    event.command = ctx.command_name().map(|name| EventCommand {
        name: name.to_string(),
        args: ctx.command_args().map(str::to_string),
    });
    event.chat_type = chat_kind(&data.update).map(String::from);
    event.chat_title = update_chat(&data.update).and_then(|chat| chat.title.clone());
    event.attention = attention(&data.update, me).into();

    match &data.update.kind {
        UpdateKind::Message(message) | UpdateKind::EditedMessage(message) => {
            let edited = matches!(data.update.kind, UpdateKind::EditedMessage(_));
            event.kind = if edited {
                "edited"
            } else if event.command.is_some() {
                "command"
            } else {
                "message"
            }
            .into();
            event.message_id = Some(message.message_id);
            event.ts = message.edit_date.unwrap_or(message.date);
            event.text = message.text.clone().or_else(|| message.caption.clone());
            event.thread_id = message.message_thread_id;
            event.reply_to = message
                .reply_to_message
                .as_ref()
                .map(|replied| EventReplyRef {
                    message_id: replied.message_id,
                    from: replied
                        .from
                        .as_ref()
                        .map(|u| u.username.clone().unwrap_or_else(|| u.first_name.clone())),
                    text: replied.text.clone(),
                    sticker: replied.sticker.as_ref().map(event_sticker),
                    media: event_media(replied),
                });
            event.sticker = message.sticker.as_ref().map(event_sticker);
            event.media = event_media(message);
        }
        UpdateKind::CallbackQuery(query) => {
            event.kind = "button".into();
            event.message_id = query.message.as_ref().map(|m| m.message_id);
            if let Some(m) = &query.message {
                event.ts = m.date;
            }
            event.text = query
                .message
                .as_ref()
                .and_then(|m| m.text.clone().or_else(|| m.caption.clone()));
            event.button = query.data.clone();
            event.command = None;
            event.thread_id = query.message.as_ref().and_then(|m| m.message_thread_id);
        }
        UpdateKind::MessageReaction(reaction) => {
            event.kind = "reaction".into();
            event.message_id = Some(reaction.message_id);
            event.text = None;
            if let Some(date) = reaction.date {
                event.ts = date;
            }
            let diff = |newer: &[botkit_telegram::ReactionType],
                        older: &[botkit_telegram::ReactionType]| {
                newer
                    .iter()
                    .filter(|r| !older.iter().any(|o| reaction_label(o) == reaction_label(r)))
                    .map(reaction_label)
                    .collect()
            };
            event.reaction = Some(EventReaction {
                added: diff(&reaction.new_reaction, &reaction.old_reaction),
                removed: diff(&reaction.old_reaction, &reaction.new_reaction),
            });
        }
        UpdateKind::Unknown => return extract_generic(ctx, event),
    }
    event
}

/// Map a Telegram `Sticker` to the event's sticker shape — also used for
/// stickers riding on a `reply_to` message.
fn event_sticker(s: &botkit_telegram::Sticker) -> EventSticker {
    EventSticker {
        file_id: s.file_id.clone(),
        emoji: s.emoji.clone(),
        set_name: s.set_name.clone(),
        format: if s.is_animated {
            "animated"
        } else if s.is_video {
            "video"
        } else {
            "static"
        }
        .into(),
        file: None,
    }
}

/// The message's first media attachment, in the same priority order for
/// top-level and `reply_to` messages.
fn event_media(message: &botkit_telegram::Message) -> Option<EventMedia> {
    message
        .video
        .as_ref()
        .map(|f| ("video", &f.file_id))
        .or_else(|| message.audio.as_ref().map(|f| ("audio", &f.file_id)))
        .or_else(|| message.voice.as_ref().map(|f| ("voice", &f.file_id)))
        .or_else(|| message.document.as_ref().map(|f| ("document", &f.file_id)))
        .or_else(|| {
            message
                .animation
                .as_ref()
                .map(|f| ("animation", &f.file_id))
        })
        .or_else(|| {
            message
                .photo
                .as_ref()
                .and_then(|sizes| sizes.last())
                .map(|p| ("photo", &p.file_id))
        })
        .map(|(kind, file_id)| EventMedia {
            kind: kind.into(),
            file_id: file_id.clone(),
            file: None,
        })
}

/// A compact label for a reaction: the emoji itself, `custom:<id>` for
/// custom emoji, `paid` for Telegram's paid reaction.
fn reaction_label(reaction: &botkit_telegram::ReactionType) -> String {
    use botkit_telegram::ReactionType;
    match reaction {
        ReactionType::Emoji { emoji } => emoji.clone(),
        ReactionType::CustomEmoji { custom_emoji_id } => format!("custom:{custom_emoji_id}"),
        ReactionType::Paid => "paid".to_string(),
    }
}

/// Fallback when the platform data is not Telegram's (or the update kind is
/// unmodelled): use whatever the unified context exposes.
fn extract_generic(ctx: &Context, mut event: ChatEvent) -> ChatEvent {
    event.command = ctx.command_name().map(|name| EventCommand {
        name: name.to_string(),
        args: ctx.command_args().map(str::to_string),
    });
    event.kind = if event.command.is_some() {
        "command"
    } else if ctx.button_id().is_some() {
        event.button = ctx.button_id().map(str::to_string);
        "button"
    } else {
        "message"
    }
    .into();
    event
}

/// The chat an update arrived in, whichever update kind it is.
fn update_chat(update: &Update) -> Option<&botkit_telegram::types::Chat> {
    match &update.kind {
        UpdateKind::Message(m) | UpdateKind::EditedMessage(m) => Some(&m.chat),
        UpdateKind::CallbackQuery(cq) => cq.message.as_ref().map(|m| &m.chat),
        UpdateKind::MessageReaction(r) => Some(&r.chat),
        UpdateKind::Unknown => None,
    }
}

/// The platform-reported chat shape, when the update carries a chat.
fn chat_kind(update: &Update) -> Option<&'static str> {
    update_chat(update).map(|chat| match chat.chat_type {
        ChatType::Private => "private",
        ChatType::Group => "group",
        ChatType::Supergroup => "supergroup",
        ChatType::Channel => "channel",
    })
}

/// `"direct"` when the update is aimed at the bot, `"ambient"` when it is
/// group chatter forwarded for context.
///
/// In private chats everything is direct. In groups the direct set is:
/// replies to the bot's own messages, `@username`/`text_mention`s of the
/// bot, bare `/command` invocations and `/command@username` aimed at the
/// bot, and button presses (keyboards only exist on the bot's own
/// messages). `/command@other` and all other room traffic is ambient:
/// the agent still sees it and may answer when it semantically continues
/// the conversation.
fn attention(update: &Update, me: &BotIdentity) -> &'static str {
    let direct = match &update.kind {
        UpdateKind::Message(m) | UpdateKind::EditedMessage(m) => {
            matches!(m.chat.chat_type, ChatType::Private) || addressed(m, me)
        }
        // Buttons only exist on messages the bot itself sent.
        UpdateKind::CallbackQuery(_) => true,
        // A reaction update does not say who authored the message it lands
        // on; outside private chats treat it as room context.
        UpdateKind::MessageReaction(r) => matches!(r.chat.chat_type, ChatType::Private),
        UpdateKind::Unknown => true,
    };
    if direct { "direct" } else { "ambient" }
}

/// Whether a group message addresses the bot: a reply to one of its
/// messages, a `/command` it should answer, or an entity mentioning its
/// username / user id.
fn addressed(message: &Message, me: &BotIdentity) -> bool {
    // A reply to one of the bot's own messages is always addressed.
    if message
        .reply_to_message
        .as_ref()
        .and_then(|replied| replied.from.as_ref())
        .is_some_and(|from| from.id == me.id)
    {
        return true;
    }
    let text = message.text.as_deref().or(message.caption.as_deref());
    let entities = message.entities.as_deref().unwrap_or(&[]);
    let mentions_me = |entity: &botkit_telegram::types::MessageEntity| -> bool {
        match entity.entity_type {
            EntityType::TextMention => entity.user.as_ref().is_some_and(|user| user.id == me.id),
            // A `mention` entity covers `@name` exactly; a `bot_command`
            // entity covers `/cmd` or `/cmd@name` — the mention target
            // trails the last `@`.
            EntityType::Mention | EntityType::BotCommand => text
                .and_then(|text| entity_slice(text, entity.offset, entity.length))
                .and_then(|slice| slice.rsplit('@').next())
                .is_some_and(|name| {
                    me.username
                        .as_deref()
                        .is_some_and(|username| name.eq_ignore_ascii_case(username))
                }),
            _ => false,
        }
    };
    if entities.iter().any(&mentions_me) {
        return true;
    }
    // The invoked command opens the message: bare `/cmd` is an explicit
    // invocation any listening bot may answer; `/cmd@other` belongs to
    // another bot.
    if let Some(command) = entities
        .iter()
        .find(|e| matches!(e.entity_type, EntityType::BotCommand) && e.offset == 0)
    {
        return text
            .and_then(|text| entity_slice(text, command.offset, command.length))
            .and_then(|slice| slice.split('@').nth(1))
            .is_none_or(|name| {
                me.username
                    .as_deref()
                    .is_some_and(|username| name.eq_ignore_ascii_case(username))
            });
    }
    false
}

/// The text an entity covers. Telegram counts `offset`/`length` in UTF-16
/// code units, so slice through the UTF-16 view.
fn entity_slice(text: &str, offset: i64, length: i64) -> Option<&str> {
    let start = utf16_boundary(text, offset)?;
    let end = utf16_boundary(text, offset.checked_add(length)?)?;
    text.get(start..end)
}

/// Byte index of the `units`-th UTF-16 code unit, or `None` when it runs
/// past the end or lands mid-character.
fn utf16_boundary(text: &str, units: i64) -> Option<usize> {
    let units = usize::try_from(units).ok()?;
    let mut seen = 0;
    for (index, ch) in text.char_indices() {
        if seen == units {
            return Some(index);
        }
        seen += ch.len_utf16();
    }
    (seen == units).then_some(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn me() -> BotIdentity {
        BotIdentity {
            id: 99,
            username: Some("mybot".to_string()),
        }
    }

    fn update(body: serde_json::Value) -> Update {
        serde_json::from_value(serde_json::json!({
            "update_id": 1,
            "message": body,
        }))
        .expect("valid update")
    }

    fn message(chat_type: &str, text: &str, entities: serde_json::Value) -> Update {
        update(serde_json::json!({
            "message_id": 1,
            "date": 0,
            "chat": { "id": -100, "type": chat_type },
            "from": { "id": 7, "is_bot": false, "first_name": "Ada" },
            "text": text,
            "entities": entities,
        }))
    }

    fn mention(offset: i64, length: i64) -> serde_json::Value {
        serde_json::json!([{ "type": "mention", "offset": offset, "length": length }])
    }

    #[test]
    fn private_messages_are_always_direct() {
        let update = message("private", "anything", serde_json::json!([]));
        assert_eq!(attention(&update, &me()), "direct");
        assert_eq!(chat_kind(&update), Some("private"));
    }

    #[test]
    fn plain_group_chatter_is_ambient() {
        let update = message("group", "morning everyone", serde_json::json!([]));
        assert_eq!(attention(&update, &me()), "ambient");
        assert_eq!(chat_kind(&update), Some("group"));
    }

    #[test]
    fn at_mention_is_direct() {
        let update = message("supergroup", "hey @mybot look", mention(4, 6));
        assert_eq!(attention(&update, &me()), "direct");
        assert_eq!(chat_kind(&update), Some("supergroup"));
    }

    #[test]
    fn mentioning_another_bot_is_ambient() {
        let update = message("group", "hey @otherbot look", mention(4, 9));
        assert_eq!(attention(&update, &me()), "ambient");
    }

    #[test]
    fn text_mention_of_the_bot_is_direct() {
        let update = message(
            "group",
            "thanks bot",
            serde_json::json!([{
                "type": "text_mention",
                "offset": 7,
                "length": 3,
                "user": { "id": 99, "is_bot": true, "first_name": "bot" }
            }]),
        );
        assert_eq!(attention(&update, &me()), "direct");
    }

    #[test]
    fn reply_to_the_bot_is_direct() {
        let update = update(serde_json::json!({
            "message_id": 2,
            "date": 0,
            "chat": { "id": -100, "type": "supergroup" },
            "from": { "id": 7, "is_bot": false, "first_name": "Ada" },
            "text": "yes",
            "reply_to_message": {
                "message_id": 1,
                "date": 0,
                "chat": { "id": -100, "type": "supergroup" },
                "from": { "id": 99, "is_bot": true, "first_name": "bot", "username": "mybot" },
                "text": "want me to check?"
            }
        }));
        assert_eq!(attention(&update, &me()), "direct");
    }

    #[test]
    fn reply_to_someone_else_is_ambient() {
        let update = update(serde_json::json!({
            "message_id": 3,
            "date": 0,
            "chat": { "id": -100, "type": "group" },
            "from": { "id": 7, "is_bot": false, "first_name": "Ada" },
            "text": "yes",
            "reply_to_message": {
                "message_id": 1,
                "date": 0,
                "chat": { "id": -100, "type": "group" },
                "from": { "id": 8, "is_bot": false, "first_name": "Grace" },
                "text": "dinner?"
            }
        }));
        assert_eq!(attention(&update, &me()), "ambient");
    }

    #[test]
    fn bare_command_in_a_group_is_direct() {
        let update = message(
            "group",
            "/start",
            serde_json::json!([{ "type": "bot_command", "offset": 0, "length": 6 }]),
        );
        assert_eq!(attention(&update, &me()), "direct");
    }

    #[test]
    fn command_aimed_at_the_bot_is_direct() {
        let update = message(
            "group",
            "/ask@mybot why",
            serde_json::json!([{ "type": "bot_command", "offset": 0, "length": 10 }]),
        );
        assert_eq!(attention(&update, &me()), "direct");
    }

    #[test]
    fn command_aimed_at_another_bot_is_ambient() {
        let update = message(
            "group",
            "/ask@otherbot why",
            serde_json::json!([{ "type": "bot_command", "offset": 0, "length": 13 }]),
        );
        assert_eq!(attention(&update, &me()), "ambient");
    }

    #[test]
    fn button_presses_are_always_direct() {
        let update: Update = serde_json::from_value(serde_json::json!({
            "update_id": 2,
            "callback_query": {
                "id": "cb1",
                "from": { "id": 7, "is_bot": false, "first_name": "Ada" },
                "chat_instance": "x",
                "data": "pick",
                "message": {
                    "message_id": 1,
                    "date": 0,
                    "chat": { "id": -100, "type": "supergroup" },
                    "text": "choose"
                }
            }
        }))
        .expect("valid update");
        assert_eq!(attention(&update, &me()), "direct");
        assert_eq!(chat_kind(&update), Some("supergroup"));
    }

    #[test]
    fn group_reactions_are_ambient() {
        let update: Update = serde_json::from_value(serde_json::json!({
            "update_id": 3,
            "message_reaction": {
                "chat": { "id": -100, "type": "group" },
                "message_id": 5,
                "user": { "id": 7, "is_bot": false, "first_name": "Ada" },
                "old_reaction": [],
                "new_reaction": [{ "type": "emoji", "emoji": "👍" }]
            }
        }))
        .expect("valid update");
        assert_eq!(attention(&update, &me()), "ambient");
    }

    #[test]
    fn utf16_offsets_slice_mentions_past_emoji() {
        // "🎉 @mybot" — the emoji is one char but two UTF-16 units, so the
        // mention entity starts at unit 3 (byte 6).
        let update = message("group", "🎉 @mybot nice", mention(3, 6));
        assert_eq!(attention(&update, &me()), "direct");
    }
}
