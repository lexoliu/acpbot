//! Chat events, keys, and the JSON envelope sent to the agent.

use std::fmt;

use serde::Serialize;

/// Identifies one conversation on one platform.
///
/// Used as the map key for per-chat senders, transcripts, and topic cells
/// in the [`ChatRouter`](crate::agent::ChatRouter) and, via
/// [`ChatKey::slug`], as the filesystem name for the chat's transcript
/// directory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChatKey {
    /// Platform tag (`telegram`, `discord`, `matrix`).
    pub platform: &'static str,
    /// Platform-native chat/channel id.
    pub id: String,
}

impl ChatKey {
    /// A filesystem-safe form of the key (`telegram-12345`).
    pub fn slug(&self) -> String {
        let id: String = self
            .id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        format!("{}-{id}", self.platform)
    }
}

impl fmt::Display for ChatKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.platform, self.id)
    }
}

/// Who triggered an event.
#[derive(Debug, Clone, Serialize)]
pub struct EventSender {
    /// Platform-native user id.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// The message a `reply` event was answering, when the platform tells us.
#[derive(Debug, Clone, Serialize)]
pub struct EventReplyRef {
    /// Id of the referenced message.
    pub message_id: i64,
    /// Display name of its author, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Its text, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A sticker the referenced message carried, when the platform
    /// includes the replied message in full.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sticker: Option<EventSticker>,
    /// Media the referenced message carried, when the platform includes
    /// the replied message in full.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<EventMedia>,
}

/// Command details for `type == "command"`.
#[derive(Debug, Clone, Serialize)]
pub struct EventCommand {
    /// Command name without prefix (`start` for `/start`).
    pub name: String,
    /// Everything after the command, when non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<String>,
}

/// A sticker attached to an inbound message — enough for the agent to
/// recognize it and resend it via `send_sticker`'s `file_id`.
#[derive(Debug, Clone, Serialize)]
pub struct EventSticker {
    /// Telegram `file_id` — resendable.
    pub file_id: String,
    /// The sticker's associated emoji, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emoji: Option<String>,
    /// Name of the sticker set it belongs to, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set_name: Option<String>,
    /// `animated` (.tgs), `video` (.webm), or `static`.
    pub format: &'static str,
    /// Local copy of the file under `<chat dir>/inbox/`, when the download
    /// succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<EventFile>,
}

/// A downloaded copy of inbound media inside the chat dir. `path` is
/// absolute so every harness's file tools resolve it the same way
/// (relative paths land wherever the harness defaults — agy's scratch
/// dir, not the session cwd); `mime` is guessed from the downloaded name.
#[derive(Debug, Clone, Serialize)]
pub struct EventFile {
    /// Absolute path (`<chat dir>/inbox/<file_unique_id>.<ext>`).
    pub path: String,
    /// MIME type guessed from the file extension.
    pub mime: String,
}

/// Non-sticker media attached to an inbound message. `file_id` resends it
/// via `send_file`; `file` points at a local copy when it downloaded.
#[derive(Debug, Clone, Serialize)]
pub struct EventMedia {
    /// `photo`, `video`, `audio`, `voice`, `document`, or `animation`.
    pub kind: &'static str,
    /// Telegram `file_id` — resendable via `send_file`.
    pub file_id: String,
    /// Local copy of the file under `<chat dir>/inbox/`, when the download
    /// succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<EventFile>,
}

/// A reaction change on a message (`type == "reaction"`): what the user
/// added and removed.
#[derive(Debug, Clone, Serialize)]
pub struct EventReaction {
    /// Reactions now present (emoji strings; custom emoji appear as
    /// `custom:<id>`).
    pub added: Vec<String>,
    /// Reactions that were removed.
    pub removed: Vec<String>,
}

/// One chat event, serialized to JSON and handed to the agent as the prompt.
///
/// The schema is the agent's whole view of every chat: it is documented
/// again in the shared `AGENTS.md` so the two never drift.
#[derive(Debug, Clone, Serialize)]
pub struct ChatEvent {
    /// `message`, `command`, `button`, `reaction`, `edited`, or `nudge`
    /// (the daemon's silence interrupt — the only kind that isn't a real
    /// platform event).
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Platform tag.
    pub platform: &'static str,
    /// Chat the event belongs to.
    pub chat: String,
    /// Event time, epoch seconds — the platform's message/edit/reaction
    /// timestamp, or arrival time for synthetic events.
    pub ts: i64,
    /// Whether the event is aimed at the bot: `"direct"` (private chat,
    /// reply to a bot message, @-mention, command, button press) or
    /// `"ambient"` (group chatter forwarded for context — answer only when
    /// it continues a conversation the bot is in or clearly concerns it).
    pub attention: &'static str,
    /// The chat's shape on the platform — `private`, `group`, `supergroup`,
    /// or `channel` — when the platform reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_type: Option<&'static str>,
    /// Message id, when the event carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<i64>,
    /// Who triggered the event.
    pub from: EventSender,
    /// Message text (or the text of the message a button was attached to).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Command details for `type == "command"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<EventCommand>,
    /// Button/callback id for `type == "button"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub button: Option<String>,
    /// The message this one replies to, when the platform reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<EventReplyRef>,
    /// A sticker the message carried.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sticker: Option<EventSticker>,
    /// Non-sticker media the message carried.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<EventMedia>,
    /// Reaction change details for `type == "reaction"`; `message_id` is the
    /// message reacted to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reaction: Option<EventReaction>,
    /// Forum topic the event arrived in, when the chat has topics. Outbound
    /// sends follow the current topic automatically.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<i64>,
}

impl ChatEvent {
    /// Render the event as the JSON text of an ACP prompt content block.
    pub fn to_prompt_text(&self) -> String {
        serde_json::to_string(self).expect("ChatEvent is always serializable")
    }

    /// Whether the event is a user telling the bot to halt — a `/stop`
    /// command aimed at the bot or a bare `stop` text. The actor
    /// intercepts these while a turn is in flight instead of prompting
    /// the agent with them; when the bot is idle they reach the agent as
    /// ordinary events. A `/stop@otherbot` (ambient command) is not ours
    /// to act on, but the bare word is — someone typing "stop" while the
    /// bot floods means it.
    pub fn is_stop(&self) -> bool {
        let stop_command = self
            .command
            .as_ref()
            .is_some_and(|command| command.name == "stop")
            && self.attention == "direct";
        let stop_text = self
            .text
            .as_deref()
            .is_some_and(|text| text.trim().eq_ignore_ascii_case("stop"));
        stop_command || stop_text
    }

    /// Every local copy this event's media/sticker downloaded to, so the
    /// actor can inline image payloads as `image` content blocks.
    pub fn files(&self) -> impl Iterator<Item = &EventFile> {
        let top = self
            .media
            .iter()
            .filter_map(|media| media.file.as_ref())
            .chain(
                self.sticker
                    .iter()
                    .filter_map(|sticker| sticker.file.as_ref()),
            );
        let replied = self.reply_to.iter().flat_map(|reply| {
            reply
                .media
                .iter()
                .filter_map(|media| media.file.as_ref())
                .chain(
                    reply
                        .sticker
                        .iter()
                        .filter_map(|sticker| sticker.file.as_ref()),
                )
        });
        top.chain(replied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A message carrying a sticker serializes its resendable metadata;
    /// absent media fields vanish entirely.
    #[test]
    fn sticker_and_media_serialize() {
        let base = ChatEvent {
            kind: "message",
            platform: "telegram",
            chat: "1".to_string(),
            ts: 1700000000,
            attention: "direct",
            chat_type: Some("private"),
            message_id: Some(9),
            from: EventSender {
                id: "7".to_string(),
                name: "Ada".to_string(),
            },
            text: None,
            command: None,
            button: None,
            reply_to: None,
            sticker: Some(EventSticker {
                file_id: "CAACAgE".to_string(),
                emoji: Some("😂".to_string()),
                set_name: Some("pack_by_bot".to_string()),
                format: "static",
                file: Some(EventFile {
                    path: "/data/chats/shared/inbox/CAACAgE.webp".to_string(),
                    mime: "image/webp".to_string(),
                }),
            }),
            media: None,
            reaction: None,
            thread_id: None,
        };
        let json = base.to_prompt_text();
        assert!(json.contains("\"file_id\":\"CAACAgE\""), "{json}");
        assert!(json.contains("\"set_name\":\"pack_by_bot\""), "{json}");
        assert!(
            json.contains("\"path\":\"/data/chats/shared/inbox/CAACAgE.webp\""),
            "{json}"
        );
        assert!(json.contains("\"mime\":\"image/webp\""), "{json}");
        assert!(!json.contains("\"media\""), "{json}");
        assert!(!json.contains("\"reaction\""), "{json}");

        let with_media = ChatEvent {
            sticker: None,
            media: Some(EventMedia {
                kind: "video",
                file_id: "BAADBQ".to_string(),
                file: None,
            }),
            ..base.clone()
        };
        let json = with_media.to_prompt_text();
        assert!(json.contains("\"kind\":\"video\""), "{json}");
        assert!(json.contains("\"file_id\":\"BAADBQ\""), "{json}");
        assert!(!json.contains("\"sticker\""), "{json}");
        // A download that never happened vanishes from the JSON entirely.
        assert!(!json.contains("\"file\""), "{json}");
        // …while `files()` surfaces the sticker's downloaded copy.
        assert_eq!(base.files().count(), 1);
        assert_eq!(with_media.files().count(), 0);

        let reacted = ChatEvent {
            kind: "reaction",
            sticker: None,
            reaction: Some(EventReaction {
                added: vec!["👍".to_string()],
                removed: vec!["custom:99".to_string()],
            }),
            ..base
        };
        let json = reacted.to_prompt_text();
        assert!(json.contains("\"type\":\"reaction\""), "{json}");
        assert!(json.contains("\"added\":[\"👍\"]"), "{json}");
        assert!(json.contains("\"removed\":[\"custom:99\"]"), "{json}");
    }

    /// `/stop` commands and bare "stop" texts halt a turn; other events —
    /// even ones containing the word — do not.
    #[test]
    fn is_stop_marks_only_real_stops() {
        let base = ChatEvent {
            kind: "message",
            platform: "telegram",
            chat: "1".to_string(),
            ts: 1700000000,
            attention: "ambient",
            chat_type: Some("supergroup"),
            message_id: Some(9),
            from: EventSender {
                id: "7".to_string(),
                name: "Ada".to_string(),
            },
            text: Some("stop".to_string()),
            command: None,
            button: None,
            reply_to: None,
            sticker: None,
            media: None,
            reaction: None,
            thread_id: None,
        };
        assert!(base.is_stop());
        assert!(
            ChatEvent {
                text: Some("  STOP ".to_string()),
                ..base.clone()
            }
            .is_stop()
        );
        assert!(
            ChatEvent {
                kind: "command",
                attention: "direct",
                text: None,
                command: Some(EventCommand {
                    name: "stop".to_string(),
                    args: None,
                }),
                ..base.clone()
            }
            .is_stop()
        );
        // …but only when the command is aimed at this bot.
        assert!(
            !ChatEvent {
                kind: "command",
                attention: "ambient",
                text: None,
                command: Some(EventCommand {
                    name: "stop".to_string(),
                    args: None,
                }),
                ..base.clone()
            }
            .is_stop()
        );
        // The word inside a sentence is chatter, not a halt.
        assert!(
            !ChatEvent {
                text: Some("don't stop now".to_string()),
                ..base.clone()
            }
            .is_stop()
        );
        assert!(!ChatEvent { text: None, ..base }.is_stop());
    }
}
