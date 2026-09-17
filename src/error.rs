//! Error types, one enum per subsystem.
//!
//! acpbot fails fast: each variant names the operation that failed, and
//! upstream errors (io, ACP, platform, sandbox) are wrapped with `#[from]`
//! rather than stringified where the caller might handle them.

use std::io;
use std::path::PathBuf;

use aither_acp::ClientError;
use botkit_core::BotError;

/// Configuration loading or validation failed.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("cannot read {path}: {source}")]
    Read {
        /// Config file path.
        path: PathBuf,
        /// Underlying IO error.
        source: io::Error,
    },
    /// The config file is not valid TOML or fails schema checks.
    #[error("cannot parse {path}: {source}")]
    Parse {
        /// Config file path.
        path: PathBuf,
        /// Underlying parse error.
        source: toml::de::Error,
    },
    /// A required environment variable is unset.
    #[error("environment variable {0} is not set")]
    MissingEnv(String),
    /// The platform section is missing required fields.
    #[error("{0}")]
    Invalid(String),
}

/// An outbound operation (send, reply, media, sticker, reaction, edit,
/// delete, pin) failed.
#[derive(Debug, thiserror::Error)]
pub enum SenderError {
    /// The platform API rejected the call.
    #[error(transparent)]
    Platform(#[from] BotError),
    /// Sticker-set publishing failed.
    #[error(transparent)]
    StickerSet(#[from] StickerSetError),
    /// Filesystem IO failed (inbox writes, file reads).
    #[error(transparent)]
    Io(#[from] io::Error),
    /// A media path has no usable filename.
    #[error("{0:?} has no usable file name")]
    NoFilename(PathBuf),
    /// The chat key's id cannot map to this platform's addressing.
    #[error("telegram chat id {0:?} is not numeric")]
    NonNumericChat(String),
    /// The target message is gone — deleted since it was sent.
    #[error("message {0} no longer exists — it was deleted")]
    MessageGone(i64),
    /// The send-channel the test `Record` platform writes to closed.
    #[cfg(test)]
    #[error("record channel closed")]
    RecordClosed,
    /// Any other send failure (test factories).
    #[cfg(test)]
    #[error("{0}")]
    Other(String),
}

/// Publishing or caching in the bot-owned sticker set failed.
#[derive(Debug, thiserror::Error)]
pub enum StickerSetError {
    /// The platform API rejected the call.
    #[error(transparent)]
    Platform(#[from] BotError),
    /// Filesystem IO failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// `published.json` could not be (de)serialized.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Telegram refused the sticker set operation for a semantic reason.
    #[error("{0}")]
    Other(String),
}

/// The sticker library (imported foreign sets) failed.
#[derive(Debug, thiserror::Error)]
pub enum StickerLibError {
    /// Sticker-set operations need the Telegram platform.
    #[error("sticker sets are only reachable on the telegram platform")]
    NoClient,
    /// The set reference could not be understood.
    #[error("{0}")]
    BadQuery(String),
    /// The catalog name is already in use.
    #[error("sticker name {0:?} is already taken")]
    NameTaken(String),
    /// The platform API rejected the call.
    #[error(transparent)]
    Telegram(#[from] BotError),
    /// Catalog file IO failed.
    #[error("cannot {action} {path}: {source}")]
    Io {
        /// What was being done (read/write/rename).
        action: &'static str,
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: io::Error,
    },
    /// `sticker_library.json` could not be parsed.
    #[error("cannot parse {path}: {source}")]
    Parse {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: serde_json::Error,
    },
    /// Serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Loading the sticker pack directory or its annotations failed.
#[derive(Debug, thiserror::Error)]
pub enum StickersError {
    /// Filesystem IO failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// `stickers.toml` could not be parsed.
    #[error("cannot parse {path}: {source}")]
    Parse {
        /// Annotation file path.
        path: PathBuf,
        /// Underlying parse error.
        source: toml::de::Error,
    },
}

/// Agent-process isolation (heel, docker, bare) failed.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// Filesystem IO failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The native sandbox could not be created.
    #[error("failed to create sandbox: {0}")]
    Create(heel::Error),
    /// The sandboxed spawn failed.
    #[error("sandboxed spawn of {program} failed: {source}")]
    Spawn {
        /// Program being spawned.
        program: String,
        /// Underlying error.
        source: heel::Error,
    },
    /// The agent command is not on PATH.
    #[error("agent command {0:?} not found on PATH")]
    NotOnPath(String),
    /// An io operation on a specific path failed.
    #[error("cannot {action} {path}: {source}")]
    PathIo {
        /// What was attempted ("create", "resolve").
        action: &'static str,
        /// The path involved.
        path: PathBuf,
        /// Underlying IO error.
        source: io::Error,
    },
    /// The sandboxed child yielded no stdio pipe.
    #[error("sandboxed child did not yield a {0} pipe")]
    MissingPipe(&'static str),
    /// No home directory could be determined.
    #[error("no home directory")]
    NoHome,
    /// Any other isolation failure.
    #[error("{0}")]
    Other(String),
}

/// The `mcp-bridge` subprocess entrypoint failed.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// Filesystem or socket IO failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// `ipc:` target used outside a heel sandbox.
    #[error("ipc: bridge target requires a heel sandbox (HEEL_IPC_ENDPOINT is unset)")]
    MissingIpcEndpoint,
    /// The heel IPC endpoint could not be reached.
    #[error("cannot reach sandbox IPC endpoint: {0}")]
    IpcConnect(heel::IpcError),
    /// An IPC call to a daemon command failed.
    #[error("ipc call to {0} failed: {1}")]
    IpcCall(String, heel::IpcError),
    /// A bare-path target on a platform without unix sockets.
    #[cfg(windows)]
    #[error(
        "unix socket target {0:?} unsupported on this platform; use tcp:HOST:PORT or ipc:COMMAND"
    )]
    UnsupportedTarget(String),
}

/// The chat MCP endpoint listener failed.
#[derive(Debug, thiserror::Error)]
pub enum McpServerError {
    /// The unix socket could not be bound.
    #[error("cannot bind {path}: {source}")]
    Bind {
        /// Socket path.
        path: PathBuf,
        /// Underlying IO error.
        source: io::Error,
    },
    /// The TCP listener could not be bound.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Top-level daemon failures.
#[derive(Debug, thiserror::Error)]
pub enum MainError {
    /// Configuration loading or validation failed.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The bridge subprocess entrypoint failed.
    #[error(transparent)]
    Bridge(#[from] BridgeError),
    /// The platform bot exited with an error.
    #[error("bot failed: {0}")]
    Bot(#[from] BotError),
    /// Filesystem IO failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The sticker library could not be loaded.
    #[error(transparent)]
    StickerLib(#[from] StickerLibError),
    /// Bad command-line usage.
    #[error("{0}")]
    Usage(String),
}

/// A shared-actor operation failed.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The ACP client call failed.
    #[error(transparent)]
    Acp(#[from] ClientError),
    /// Filesystem IO failed (chat dir, AGENTS.md, mcp config, media).
    #[error(transparent)]
    Io(#[from] io::Error),
    /// Process isolation failed.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// Sender construction failed.
    #[error(transparent)]
    Sender(#[from] SenderError),
    /// The MCP listener failed.
    #[error(transparent)]
    McpServer(#[from] McpServerError),
    /// JSON (de)serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Any other actor failure.
    #[error("{0}")]
    Other(String),
}
