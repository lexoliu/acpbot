//! Daemon configuration, loaded from a TOML file.

use std::path::PathBuf;

use serde::Deserialize;

use crate::error::ConfigError;

/// Top-level `acpbot.toml`.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Chat platform settings.
    pub platform: PlatformConfig,
    /// ACP agent settings.
    #[serde(default)]
    pub agent: AgentConfig,
    /// Filesystem locations.
    #[serde(default)]
    pub paths: PathsConfig,
    /// Optional persona text injected into the shared `AGENTS.md`.
    pub persona: Option<String>,
    /// The daemon-hosted stealth browser (`[browser]`).
    #[serde(default)]
    pub browser: BrowserConfig,
}

/// Which chat platform to connect to, and its credentials.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum PlatformConfig {
    /// Telegram over long polling.
    Telegram {
        /// Bot token. Prefer `token_env` so the secret stays out of the file.
        token: Option<String>,
        /// Environment variable holding the bot token.
        token_env: Option<String>,
        /// Telegram sticker-set name. Default `acpbot_by_<bot username>` —
        /// Telegram requires the `_by_<bot>` suffix. Only override to adopt
        /// an existing set this bot already manages.
        sticker_set_name: Option<String>,
        /// Sticker-set display title, used when the set is created.
        /// Default "acpbot stickers".
        sticker_set_title: Option<String>,
        /// Telegram user id who owns the sticker set — sticker sets are
        /// owned by humans; the bot manages it on their behalf. When unset,
        /// the first private-chat user to trigger a sticker send becomes
        /// the owner (a private chat's id is its user's id).
        sticker_set_owner: Option<i64>,
    },
    /// A CLI-driven platform for local debugging: no credentials, events
    /// arrive over the JSONL wire protocol.
    Cli {
        /// Unix socket the bot listens on. Drivers (the `botkit-cli`
        /// binary, an agent) connect here. Mutually exclusive with `stdio`.
        socket: Option<PathBuf>,
        /// Drive the bot over stdin/stdout instead of a socket.
        stdio: Option<bool>,
    },
    /// Discord over the gateway websocket.
    Discord {
        /// Bot token. Prefer `token_env` so the secret stays out of the file.
        token: Option<String>,
        /// Environment variable holding the bot token.
        token_env: Option<String>,
        /// The application's id, from the Discord developer portal. Not a
        /// secret; safe to commit.
        application_id: String,
    },
}

impl PlatformConfig {
    /// The platform tag events and `platform:id` chat keys carry —
    /// the `kind` spelling.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Telegram { .. } => "telegram",
            Self::Cli { .. } => "cli",
            Self::Discord { .. } => "discord",
        }
    }

    /// Resolve the Telegram bot token from `token_env` or `token`.
    ///
    /// # Errors
    /// [`ConfigError::MissingEnv`] when `token_env` names an unset variable,
    /// [`ConfigError::Invalid`] when neither token field is present or the
    /// platform is not Telegram.
    pub fn telegram_token(&self) -> Result<String, ConfigError> {
        let Self::Telegram {
            token, token_env, ..
        } = self
        else {
            return Err(ConfigError::Invalid(
                "telegram_token on a non-telegram platform".to_string(),
            ));
        };
        if let Some(var) = token_env {
            return std::env::var(var).map_err(|_| ConfigError::MissingEnv(var.clone()));
        }
        token.clone().ok_or_else(|| {
            ConfigError::Invalid("telegram config needs `token` or `token_env`".to_string())
        })
    }

    /// Resolve the Discord bot token from `token_env` or `token`.
    ///
    /// # Errors
    /// [`ConfigError::MissingEnv`] when `token_env` names an unset variable,
    /// [`ConfigError::Invalid`] when neither token field is present or the
    /// platform is not Discord.
    pub fn discord_token(&self) -> Result<String, ConfigError> {
        let Self::Discord {
            token, token_env, ..
        } = self
        else {
            return Err(ConfigError::Invalid(
                "discord_token on a non-discord platform".to_string(),
            ));
        };
        if let Some(var) = token_env {
            return std::env::var(var).map_err(|_| ConfigError::MissingEnv(var.clone()));
        }
        token.clone().ok_or_else(|| {
            ConfigError::Invalid("discord config needs `token` or `token_env`".to_string())
        })
    }

    /// The Discord application id.
    ///
    /// # Errors
    /// [`ConfigError::Invalid`] when the platform is not Discord.
    pub fn discord_application_id(&self) -> Result<String, ConfigError> {
        let Self::Discord { application_id, .. } = self else {
            return Err(ConfigError::Invalid(
                "discord_application_id on a non-discord platform".to_string(),
            ));
        };
        Ok(application_id.clone())
    }

    /// The CLI transport the platform section configures.
    ///
    /// # Errors
    /// [`ConfigError::Invalid`] when neither `socket` nor `stdio` is set, or
    /// the platform is not CLI.
    pub fn cli_transport(&self) -> Result<botkit_cli::Transport, ConfigError> {
        let Self::Cli { socket, stdio } = self else {
            return Err(ConfigError::Invalid(
                "cli_transport on a non-cli platform".to_string(),
            ));
        };
        if stdio.unwrap_or(false) {
            return Ok(botkit_cli::Transport::Stdio);
        }
        match socket {
            Some(path) => cli_socket_transport(path),
            None => Err(ConfigError::Invalid(
                "cli platform needs `socket` or `stdio = true`".to_string(),
            )),
        }
    }

    /// The sticker-set options for a Telegram platform.
    /// `(name, title, owner user id)` — all optional.
    pub fn telegram_sticker_set(&self) -> (Option<String>, Option<String>, Option<i64>) {
        match self {
            Self::Telegram {
                sticker_set_name,
                sticker_set_title,
                sticker_set_owner,
                ..
            } => (
                sticker_set_name.clone(),
                sticker_set_title.clone(),
                *sticker_set_owner,
            ),
            Self::Cli { .. } | Self::Discord { .. } => (None, None, None),
        }
    }
}

/// The CLI `socket` transport — unix sockets exist only on unix; on Windows
/// the `botkit-cli` wire runs over stdio only.
#[cfg(unix)]
fn cli_socket_transport(path: &std::path::Path) -> Result<botkit_cli::Transport, ConfigError> {
    Ok(botkit_cli::Transport::Unix(path.to_path_buf()))
}

/// The CLI `socket` transport — unix sockets exist only on unix; on Windows
/// the `botkit-cli` wire runs over stdio only.
#[cfg(windows)]
fn cli_socket_transport(_path: &std::path::Path) -> Result<botkit_cli::Transport, ConfigError> {
    Err(ConfigError::Invalid(
        "cli `socket` is unix-only; use `stdio = true`".to_string(),
    ))
}

/// How to launch and configure the ACP agent process.
#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    /// Program to spawn (default `devin`).
    #[serde(default = "default_command")]
    pub command: String,
    /// Arguments for the program (default `["acp"]`).
    #[serde(default = "default_args")]
    pub args: Vec<String>,
    /// Session mode to activate after session setup (default `bypass`).
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Model config option (default `swe-2-medium`).
    #[serde(default = "default_model")]
    pub model: String,
    /// How the agent process is isolated (default `native`).
    #[serde(default)]
    pub isolation: AgentIsolation,
    /// Seconds of event silence after which the daemon asks the agent to
    /// compact the session once (default 1800 — comfortably inside the
    /// ~1h window prompt caches survive, so the compaction itself is
    /// cheap). `0` disables idle compaction.
    #[serde(default = "default_idle_compact_secs")]
    pub idle_compact_secs: u64,
    /// Seconds of outbound silence inside a turn before the daemon cancels
    /// the turn and nudges the agent to message the user (default 10).
    /// `0` disables the nudge.
    #[serde(default = "default_nudge_after_secs")]
    pub nudge_after_secs: u64,
}

/// How the agent process is isolated from the host.
///
/// Network access is never restricted: the agent and its MCP bridge must
/// reach the platform APIs and the daemon regardless of mode.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum AgentIsolation {
    /// Native OS sandbox via `heel` (default): the filesystem is locked to
    /// explicit grants, credentials and user data are protected, and network
    /// traffic is unrestricted.
    Native(NativeIsolation),
    /// Run the agent inside a Docker container. The image must contain the
    /// agent `command` and `acpbot` (for `mcp-bridge`) on `PATH`.
    Docker(DockerIsolation),
    /// Run the agent directly on the host with no isolation.
    #[serde(rename = "none")]
    Bare,
}

impl Default for AgentIsolation {
    fn default() -> Self {
        Self::Native(NativeIsolation::default())
    }
}

/// Native-sandbox settings (`kind = "native"`).
#[derive(Debug, Deserialize)]
pub struct NativeIsolation {
    /// Host environment variables forwarded into the sandbox (API keys the
    /// agent CLI reads, `TERM`, ...). `HOME` is always provided.
    #[serde(default)]
    pub env_passthrough: Vec<String>,
    /// Agent state directories granted read+write (tilde-expanded, created
    /// when missing). Default `~/.devin` and `~/.config/devin`.
    #[serde(default = "default_state_dirs")]
    pub state_dirs: Vec<PathBuf>,
    /// Agent install directories granted read+write+execute, for toolchains
    /// that self-update in place (default `~/.local/share/devin`, which holds
    /// devin's versions, logs, and session database).
    #[serde(default = "default_runtime_dirs")]
    pub runtime_dirs: Vec<PathBuf>,
    /// Extra filesystem grants for agent toolchains (e.g. a node_modules
    /// install or a language runtime directory).
    #[serde(default)]
    pub grants: Vec<GrantSpec>,
}

/// One filesystem grant inside the native sandbox.
#[derive(Debug, Deserialize)]
pub struct GrantSpec {
    /// Path to grant (tilde-expanded).
    pub path: PathBuf,
    /// `r`, `rw`, `rx`, or `rwx`.
    pub access: GrantAccess,
}

/// Access level of a [`GrantSpec`].
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrantAccess {
    /// Read only.
    R,
    /// Read and write.
    Rw,
    /// Read and execute.
    Rx,
    /// Read, write, and execute.
    Rwx,
}

/// Home-relative paths for the built-in defaults. `Config::load` still
/// tilde-expands whatever the file supplies.
fn home_dirs<const N: usize>(names: [&str; N]) -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    names.iter().map(|name| home.join(name)).collect()
}

fn default_state_dirs() -> Vec<PathBuf> {
    home_dirs([".devin", ".config/devin"])
}

fn default_runtime_dirs() -> Vec<PathBuf> {
    home_dirs([".local/share/devin"])
}

impl Default for NativeIsolation {
    /// The serde defaults, applied also when constructing in code: an
    /// isolation default that grants nothing cannot run the agent at all.
    fn default() -> Self {
        Self {
            env_passthrough: Vec::new(),
            state_dirs: default_state_dirs(),
            runtime_dirs: default_runtime_dirs(),
            grants: Vec::new(),
        }
    }
}

/// Docker settings (`kind = "docker"`).
#[derive(Debug, Deserialize)]
pub struct DockerIsolation {
    /// Image tag containing the agent command and the bridge command below.
    pub image: String,
    /// Home directory inside the container (where agent state mounts land).
    /// Default `/root`.
    #[serde(default = "default_container_home")]
    pub home: String,
    /// Argv prefix inside the container that runs `acpbot mcp-bridge`; the
    /// generated `mcp_config.json` appends the tcp target to it. Default
    /// `["acpbot", "mcp-bridge"]`.
    #[serde(default = "default_bridge_command")]
    pub bridge_command: Vec<String>,
    /// Extra `docker run` flags, e.g. `"--memory=2g"`.
    #[serde(default)]
    pub args: Vec<String>,
}

fn default_container_home() -> String {
    "/root".to_string()
}

fn default_bridge_command() -> Vec<String> {
    vec!["acpbot".to_string(), "mcp-bridge".to_string()]
}

fn default_command() -> String {
    "devin".to_string()
}

fn default_args() -> Vec<String> {
    vec!["acp".to_string()]
}

fn default_mode() -> String {
    "bypass".to_string()
}

fn default_model() -> String {
    "swe-2-medium".to_string()
}

fn default_idle_compact_secs() -> u64 {
    30 * 60
}

fn default_nudge_after_secs() -> u64 {
    10
}

impl AgentConfig {
    /// The quiet stretch that triggers one idle compaction.
    pub fn idle_compact(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.idle_compact_secs)
    }

    /// The outbound silence that triggers a nudge mid-turn.
    pub fn nudge_after(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.nudge_after_secs)
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            command: default_command(),
            args: default_args(),
            mode: default_mode(),
            model: default_model(),
            isolation: AgentIsolation::default(),
            idle_compact_secs: default_idle_compact_secs(),
            nudge_after_secs: default_nudge_after_secs(),
        }
    }
}

/// The daemon-hosted browser (`[browser]`).
///
/// One real Chrome instance, driven by the daemon over CDP and shared by
/// the agent and all its subagents through the `browser_*` tools on the
/// chat MCP endpoint — so it works under every isolation mode and
/// outlives per-turn agent respawns. Launch flags are minimal and
/// automation-free (`--enable-automation` is never passed, and the CDP
/// client never sends `Runtime.enable`), which is what makes the
/// browser read as a human's to anti-bot walls.
#[derive(Debug, Clone, Deserialize)]
pub struct BrowserConfig {
    /// `false` runs the daemon with no browser tools (default `true`).
    #[serde(default = "default_browser_enabled")]
    pub enabled: bool,
    /// Chrome-family executable. Default: auto-detect Google Chrome /
    /// Chromium / Edge / Brave from the usual install paths.
    pub executable: Option<PathBuf>,
    /// Run headless (`--headless=new`, default `true`). Set `false` on a
    /// machine with a logged-in desktop for the most human-looking
    /// browser — a headed window is the strongest fingerprint signal.
    #[serde(default = "default_browser_headless")]
    pub headless: bool,
    /// Browser profile directory (cookies, logins). Default
    /// `<data_dir>/browser-profile`; persists across restarts.
    pub profile_dir: Option<PathBuf>,
    /// Extra Chrome command-line arguments appended after the built-in
    /// minimal set (e.g. a `--proxy-server`, `--lang`, window size).
    /// Automation flags like `--enable-automation` defeat the point —
    /// don't add them.
    #[serde(default)]
    pub args: Vec<String>,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enabled: default_browser_enabled(),
            executable: None,
            headless: default_browser_headless(),
            profile_dir: None,
            args: Vec::new(),
        }
    }
}

fn default_browser_enabled() -> bool {
    true
}

fn default_browser_headless() -> bool {
    true
}

/// Directories the daemon owns.
#[derive(Debug, Deserialize)]
pub struct PathsConfig {
    /// Root for sockets, chat working dirs, the session registry, and
    /// transcripts. Default `~/.local/share/acpbot`.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    /// Sticker pack directory. Default `<data_dir>/stickers`.
    pub stickers: Option<PathBuf>,
}

fn default_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("acpbot")
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            stickers: None,
        }
    }
}

impl PathsConfig {
    /// The sticker pack directory (configured or the default under `data_dir`).
    pub fn sticker_dir(&self) -> PathBuf {
        self.stickers
            .clone()
            .unwrap_or_else(|| self.data_dir.join("stickers"))
    }
}

/// Expand a leading `~` in a path against the user's home directory.
pub fn expand_tilde(path: &mut PathBuf) {
    if let Ok(text) = path.clone().into_os_string().into_string()
        && let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        *path = home.join(rest);
    }
}

impl Config {
    /// Load and normalize the config file.
    ///
    /// # Errors
    /// [`ConfigError::Read`] when the file cannot be read,
    /// [`ConfigError::Parse`] when it is not valid TOML.
    pub fn load(path: &std::path::Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let mut config: Self = toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        expand_tilde(&mut config.paths.data_dir);
        if let Some(stickers) = &mut config.paths.stickers {
            expand_tilde(stickers);
        }
        if let AgentIsolation::Native(native) = &mut config.agent.isolation {
            for path in native
                .state_dirs
                .iter_mut()
                .chain(native.runtime_dirs.iter_mut())
                .chain(native.grants.iter_mut().map(|grant| &mut grant.path))
            {
                expand_tilde(path);
            }
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_isolation_is_native_with_devin_dirs() {
        let AgentIsolation::Native(native) = AgentIsolation::default() else {
            panic!("default isolation must be native");
        };
        let home = dirs::home_dir().expect("home dir");
        assert!(native.state_dirs.contains(&home.join(".devin")));
        assert!(
            native
                .runtime_dirs
                .contains(&home.join(".local/share/devin"))
        );
    }

    #[test]
    fn isolation_variants_parse() {
        #[derive(Deserialize)]
        struct Wrapper {
            isolation: AgentIsolation,
        }

        let parsed: Wrapper =
            toml::from_str(r#"isolation = { kind = "none" }"#).expect("bare isolation parses");
        assert!(matches!(parsed.isolation, AgentIsolation::Bare));

        let parsed: Wrapper =
            toml::from_str(r#"isolation = { kind = "docker", image = "agent:dev" }"#)
                .expect("docker isolation parses");
        let AgentIsolation::Docker(docker) = parsed.isolation else {
            panic!("expected docker isolation");
        };
        assert_eq!(docker.image, "agent:dev");
        assert_eq!(docker.home, "/root");
        assert_eq!(docker.bridge_command, ["acpbot", "mcp-bridge"]);
    }

    #[test]
    fn example_config_parses() {
        // Keeps the documented example honest — a field rename that breaks
        // parsing fails here, not on a user's first run.
        toml::from_str::<Config>(include_str!("../acpbot.example.toml"))
            .expect("acpbot.example.toml must parse");
    }
}
