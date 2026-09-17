//! The ACP client-side handler: what the agent says back to us.
//!
//! Session updates (message chunks, tool calls, plans) are appended to the
//! chat's `transcript.log` and logged via `tracing`. Nothing the agent writes
//! as plain text is delivered to the chat — output goes through the `chat`
//! MCP tools only.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aither_acp::{
    ClientHandler, RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult,
    SessionNotification, SessionUpdate,
};
use aither_mcp::protocol::JsonRpcError;
use futures_lite::io::AsyncWriteExt;
use tracing::{debug, warn};

/// Handles agent-to-client traffic for one chat session.
#[derive(Debug)]
pub struct BotClientHandler {
    /// File the session's updates are appended to, one JSON line each.
    transcript: PathBuf,
    /// Set when the session's `available_commands_update` advertises a
    /// `compact` command — the actor reads it to decide whether an idle
    /// compaction is safe to send (`/compact` is devin-specific; other
    /// harnesses would treat it as ordinary text).
    compact_supported: Arc<AtomicBool>,
}

impl BotClientHandler {
    /// A handler appending to `transcript`, publishing command support
    /// flags onto `compact_supported`.
    pub fn new(transcript: PathBuf, compact_supported: Arc<AtomicBool>) -> Self {
        Self {
            transcript,
            compact_supported,
        }
    }
}

impl ClientHandler for BotClientHandler {
    async fn session_update(&self, notification: SessionNotification) {
        let update = &notification.update;
        debug!(?update, "session update");

        if let SessionUpdate::AvailableCommandsUpdate(commands) = update {
            self.compact_supported.store(
                commands
                    .available_commands
                    .iter()
                    .any(|command| command.name == "compact"),
                Ordering::Relaxed,
            );
        }

        let Ok(mut line) = serde_json::to_string(update) else {
            return;
        };
        line.push('\n');
        match async_fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.transcript)
            .await
        {
            Ok(mut file) => {
                if let Err(error) = file.write_all(line.as_bytes()).await {
                    warn!(%error, "failed to append transcript");
                }
            }
            Err(error) => warn!(%error, "failed to open transcript"),
        }
    }

    async fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> Result<RequestPermissionResult, JsonRpcError> {
        // Sessions run in bypass mode, so this is a last-resort path: pick the
        // most permissive-looking option rather than cancelling the turn.
        let option_id = params
            .options
            .iter()
            .find(|o| {
                let id = o.option_id.to_ascii_lowercase();
                id.contains("allow") || id.contains("proceed") || id.contains("yes")
            })
            .or_else(|| params.options.first())
            .map_or_else(|| "allow".to_string(), |o| o.option_id.clone());
        Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Selected { option_id },
            meta: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_acp::{AvailableCommand, AvailableCommandsUpdate, SessionUpdate};
    use futures_lite::future::block_on;

    fn notification(names: &[&str]) -> SessionNotification {
        SessionNotification {
            session_id: "s".to_string(),
            update: SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate {
                available_commands: names
                    .iter()
                    .map(|name| AvailableCommand {
                        name: (*name).to_string(),
                        description: String::new(),
                        input: None,
                        meta: None,
                    })
                    .collect(),
                meta: None,
            }),
            meta: None,
            extra: Default::default(),
        }
    }

    /// The `compact_supported` flag tracks whatever the session last
    /// advertised — a harness without `compact` must not get `/compact`.
    #[test]
    fn compact_flag_follows_advertised_commands() {
        let flag = Arc::new(AtomicBool::new(false));
        let handler = BotClientHandler::new(
            PathBuf::from("/tmp/acpbot-test-transcript.log"),
            flag.clone(),
        );

        block_on(handler.session_update(notification(&["login", "ask"])));
        assert!(!flag.load(Ordering::Relaxed));

        block_on(handler.session_update(notification(&["login", "compact"])));
        assert!(flag.load(Ordering::Relaxed));
    }
}
