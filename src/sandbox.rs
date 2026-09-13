//! How one chat's agent process is launched and isolated.
//!
//! Three runtimes, selected by `[agent.isolation]`:
//!
//! - `native` (default): a [`heel`] sandbox per chat — strict filesystem and
//!   credential protections, unrestricted network (`AllowAll` policy, so
//!   traffic flows through heel's local proxy with nothing denied), and the
//!   chat MCP tools carried over heel's own IPC channel: the sandboxed
//!   `mcp-bridge` relays each JSON-RPC line through `chat-mcp`, which a
//!   host-side relay task pumps into the chat's unix socket.
//! - `docker`: `docker run` wraps the agent command; the chat directory and
//!   the agent state directories are bind-mounted, and the bridge reaches
//!   the daemon over `host.docker.internal` TCP.
//! - `none`: a direct host spawn, kept for debugging and harnesses that
//!   cannot run under either boundary.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use crate::error::SandboxError;
use aither_acp::AcpClient;
use aither_mcp::transport::StreamTransport;
use async_channel::Sender as ChanSender;
use futures_lite::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use heel::{
    Access, AllowAll, Child, IpcCommand, IpcRouter, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig,
};
use serde::Deserialize;
use tracing::warn;

use crate::config::{AgentConfig, AgentIsolation, DockerIsolation, GrantAccess, NativeIsolation};
use crate::handler::BotClientHandler;
use crate::mcpserver::ChatEndpoint;

/// The IPC command the sandboxed `mcp-bridge` invokes for each JSON-RPC line.
pub const CHAT_MCP_IPC: &str = "chat-mcp";

/// The bridge target the chat's `mcp_config.json` should advertise.
#[derive(Debug, Clone)]
pub enum BridgeTarget {
    /// `acpbot mcp-bridge <path>` — a unix socket on the host.
    #[cfg(unix)]
    Unix(PathBuf),
    /// `acpbot mcp-bridge tcp:host.docker.internal:<port>` — the daemon's
    /// per-chat loopback listener, reached from inside the container.
    DockerTcp(u16),
    /// `acpbot mcp-bridge tcp:127.0.0.1:<port>` — the host-side endpoint on
    /// platforms without unix sockets (Windows `bare`/`native`).
    Tcp(std::net::SocketAddr),
    /// `acpbot mcp-bridge ipc:chat-mcp` — relayed through the sandbox's IPC.
    Ipc,
}

impl BridgeTarget {
    /// The argument `mcp-bridge` is invoked with.
    pub fn arg(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::Unix(path) => path.to_string_lossy().into_owned(),
            Self::DockerTcp(port) => format!("tcp:host.docker.internal:{port}"),
            Self::Tcp(addr) => format!("tcp:{addr}"),
            Self::Ipc => format!("ipc:{CHAT_MCP_IPC}"),
        }
    }

    /// The bridge target matching an isolation mode over `endpoint`: the
    /// socket path for `bare`, docker's gateway TCP for `docker`, and the
    /// IPC relay for `native`.
    pub fn for_isolation(isolation: &AgentIsolation, endpoint: &ChatEndpoint) -> Self {
        match isolation {
            AgentIsolation::Native(_) => Self::Ipc,
            AgentIsolation::Docker(_) => match endpoint {
                ChatEndpoint::Tcp(addr) => Self::DockerTcp(addr.port()),
                #[cfg(unix)]
                ChatEndpoint::Unix(_) => unreachable!("docker endpoint is always tcp"),
            },
            AgentIsolation::Bare => match endpoint {
                #[cfg(unix)]
                ChatEndpoint::Unix(path) => Self::Unix(path.clone()),
                ChatEndpoint::Tcp(addr) => Self::Tcp(*addr),
            },
        }
    }
}

/// Everything one chat needs to (re)spawn its agent process.
pub enum AgentRuntime {
    /// A direct host spawn (`kind = "none"`).
    Bare,
    /// A `heel` sandbox (`kind = "native"`).
    Native(Box<NativeRuntime>),
    /// A `docker run` wrapper (`kind = "docker"`).
    Docker(DockerRuntime),
}

/// State the native runtime keeps: the sandbox itself (whose drop kills
/// everything inside) and the resolved agent program.
pub struct NativeRuntime {
    sandbox: Sandbox<AllowAll>,
    /// Canonical path of the agent binary — the sandbox's curated `PATH`
    /// does not carry host install locations, so spawns name it directly.
    program: String,
}

/// Precomputed `docker run` arguments for one chat.
pub struct DockerRuntime {
    /// Everything between `docker run` and the agent command: mounts, env,
    /// the caller's extra args, and the image tag (which must come last
    /// before the command).
    run_args: Vec<String>,
}

/// A freshly spawned agent process.
pub struct SpawnedAgent {
    /// The connected ACP client.
    pub client: AcpClient<BotClientHandler>,
    /// The connection driver; spawn it on the executor.
    pub connection: Pin<Box<dyn Future<Output = ()> + Send>>,
    /// The sandboxed child handle, kept so a respawn can kill the process
    /// explicitly rather than waiting for the pipes to close. `None` for
    /// bare and docker spawns, whose connection driver owns the process.
    pub child: Option<Child>,
}

impl AgentRuntime {
    /// Build the runtime for one chat. For `native` this creates the sandbox
    /// (working directory, proxy, IPC server); for `docker` it assembles the
    /// `docker run` argument list.
    pub async fn create(
        isolation: &AgentIsolation,
        agent: &AgentConfig,
        cwd: &Path,
        endpoint: &ChatEndpoint,
        bridge_bin: &Path,
        sticker_dir: &Path,
    ) -> Result<Self, SandboxError> {
        match isolation {
            AgentIsolation::Bare => Ok(Self::Bare),
            AgentIsolation::Native(native) => {
                NativeRuntime::create(native, agent, cwd, endpoint, bridge_bin, sticker_dir)
                    .await
                    .map(|runtime| Self::Native(Box::new(runtime)))
            }
            AgentIsolation::Docker(docker) => {
                DockerRuntime::create(docker, cwd, sticker_dir).map(Self::Docker)
            }
        }
    }

    /// Spawn the agent and connect over its stdio pipes.
    pub async fn spawn_agent(
        &self,
        agent: &AgentConfig,
        cwd: &Path,
        handler: BotClientHandler,
    ) -> Result<SpawnedAgent, SandboxError> {
        match self {
            Self::Bare => spawn_stdio(&agent.command, agent.args.clone(), cwd, handler),
            Self::Docker(docker) => {
                let args = docker
                    .run_args
                    .iter()
                    .cloned()
                    .chain(std::iter::once(agent.command.clone()))
                    .chain(agent.args.iter().cloned())
                    .collect();
                spawn_stdio("docker", args, cwd, handler)
            }
            Self::Native(native) => native.spawn(agent, cwd, handler).await,
        }
    }
}

impl NativeRuntime {
    async fn create(
        native: &NativeIsolation,
        agent: &AgentConfig,
        cwd: &Path,
        endpoint: &ChatEndpoint,
        bridge_bin: &Path,
        sticker_dir: &Path,
    ) -> Result<Self, SandboxError> {
        let program = resolve_program(&agent.command)?;
        let bridge_bin =
            std::fs::canonicalize(bridge_bin).map_err(|source| SandboxError::PathIo {
                action: "resolve",
                path: bridge_bin.to_path_buf(),
                source,
            })?;

        let relay = spawn_relay(endpoint.clone());
        let router = IpcRouter::new().register(McpRelay { tx: relay });

        // The sticker pack is deliberately writable: the agent evolves it —
        // new memes land as files plus a stickers.toml entry and are
        // sendable immediately (the daemon re-scans the dir per call).
        std::fs::create_dir_all(sticker_dir).map_err(|source| SandboxError::PathIo {
            action: "create",
            path: sticker_dir.to_path_buf(),
            source,
        })?;

        let mut builder = SandboxConfig::builder()
            .network(AllowAll)
            .security(SecurityConfig::strict())
            .filesystem_strict(true)
            .working_dir(cwd)
            .grant(&program, Access::EXEC)
            .grant(&bridge_bin, Access::EXEC)
            .grant(sticker_dir, Access::WRITE)
            .env_passthroughs(native.env_passthrough.iter().cloned())
            .ipc(router);

        // Agent state directories hold the harness's auth and session store;
        // runtime directories are install trees the agent writes and executes
        // (self-updating toolchains). Both are created host-side first so the
        // grants can canonicalize.
        for dir in &native.state_dirs {
            std::fs::create_dir_all(dir).map_err(|source| SandboxError::PathIo {
                action: "create",
                path: dir.clone(),
                source,
            })?;
            builder = builder.grant(dir, Access::WRITE);
        }
        for dir in &native.runtime_dirs {
            std::fs::create_dir_all(dir).map_err(|source| SandboxError::PathIo {
                action: "create",
                path: dir.clone(),
                source,
            })?;
            builder = builder.grant(dir, Access::WRITE | Access::EXEC);
        }
        for grant in &native.grants {
            builder = builder.grant(&grant.path, grant.access.into());
        }

        let sandbox = Sandbox::with_config(builder.build())
            .await
            .map_err(SandboxError::Create)?;

        Ok(Self {
            sandbox,
            program: program.to_string_lossy().into_owned(),
        })
    }

    async fn spawn(
        &self,
        agent: &AgentConfig,
        cwd: &Path,
        handler: BotClientHandler,
    ) -> Result<SpawnedAgent, SandboxError> {
        let home = dirs::home_dir().ok_or(SandboxError::NoHome)?;
        let mut child = self
            .sandbox
            .command(self.program.clone())
            .args(agent.args.iter())
            .env("HOME", home.to_string_lossy())
            .current_dir(cwd)
            .stdin(StdioConfig::Piped)
            .stdout(StdioConfig::Piped)
            // Agent diagnostics stay visible in the daemon's own logs.
            .stderr(StdioConfig::Inherit)
            .spawn()
            .await
            .map_err(|source| SandboxError::Spawn {
                program: self.program.clone(),
                source,
            })?;

        let stdout = child
            .take_stdout()
            .ok_or(SandboxError::MissingPipe("stdout"))?;
        let stdin = child
            .take_stdin()
            .ok_or(SandboxError::MissingPipe("stdin"))?;
        let transport = StreamTransport::new(
            BufReader::new(async_fs::File::from(stdout)),
            async_fs::File::from(stdin),
        );
        let (client, connection) = AcpClient::connect(transport, handler);
        Ok(SpawnedAgent {
            client,
            connection: Box::pin(connection),
            child: Some(child),
        })
    }
}

impl DockerRuntime {
    fn create(
        docker: &DockerIsolation,
        cwd: &Path,
        sticker_dir: &Path,
    ) -> Result<Self, SandboxError> {
        let host_home = dirs::home_dir().ok_or(SandboxError::NoHome)?;
        let cwd = std::fs::canonicalize(cwd).map_err(|source| SandboxError::PathIo {
            action: "resolve",
            path: cwd.to_path_buf(),
            source,
        })?;
        let sticker_dir =
            std::fs::canonicalize(sticker_dir).map_err(|source| SandboxError::PathIo {
                action: "resolve",
                path: sticker_dir.to_path_buf(),
                source,
            })?;

        let mut run_args = vec![
            "run".to_string(),
            "--rm".to_string(),
            "-i".to_string(),
            // Lets `host.docker.internal` resolve on native-Linux docker;
            // Docker Desktop provides it already and the flag is harmless.
            "--add-host".to_string(),
            "host.docker.internal:host-gateway".to_string(),
            "-v".to_string(),
            format!("{0}:{0}", cwd.display()),
            "-w".to_string(),
            cwd.to_string_lossy().into_owned(),
            "-e".to_string(),
            format!("HOME={}", docker.home),
        ];

        // The harness's state directories are mounted so its auth and ACP
        // session store persist across containers and runtimes. The sticker
        // pack is mounted too so the agent can extend it.
        for dir in [".devin", ".config/devin"] {
            let host_dir = host_home.join(dir);
            std::fs::create_dir_all(&host_dir).map_err(|source| SandboxError::PathIo {
                action: "create",
                path: host_dir.clone(),
                source,
            })?;
            run_args.push("-v".to_string());
            run_args.push(format!("{}:{}/{}", host_dir.display(), docker.home, dir));
        }
        run_args.push("-v".to_string());
        run_args.push(format!("{0}:{0}", sticker_dir.display()));

        run_args.extend(docker.args.iter().cloned());
        run_args.push(docker.image.clone());
        Ok(Self { run_args })
    }
}

/// Spawn through `async_process` — used directly for `none` and as the
/// `docker run` wrapper for `docker`.
fn spawn_stdio(
    program: &str,
    args: Vec<String>,
    cwd: &Path,
    handler: BotClientHandler,
) -> Result<SpawnedAgent, SandboxError> {
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let (client, connection) = AcpClient::spawn(
        program,
        &arg_refs,
        std::iter::empty::<(String, String)>(),
        cwd.to_path_buf(),
        handler,
    )
    .map_err(|e| SandboxError::Other(format!("failed to spawn {program}: {e}")))?;
    Ok(SpawnedAgent {
        client,
        connection: Box::pin(connection),
        child: None,
    })
}

/// Resolve the agent command to a canonical absolute path.
///
/// Sandboxed children get a curated `PATH` that does not include the host's
/// install locations, so a native spawn must name the binary itself; a
/// bare spawn keeps the configured name and lets the shell environment
/// resolve it.
fn resolve_program(command: &str) -> Result<PathBuf, SandboxError> {
    let candidate = if command.contains('/') {
        PathBuf::from(command)
    } else {
        std::env::var_os("PATH")
            .into_iter()
            .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
            .map(|dir| dir.join(command))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| SandboxError::NotOnPath(command.to_string()))?
    };
    std::fs::canonicalize(&candidate).map_err(|source| SandboxError::PathIo {
        action: "resolve",
        path: candidate.clone(),
        source,
    })
}

/// One JSON-RPC line from the sandboxed bridge, plus where its answer goes.
struct RelayRequest {
    line: String,
    reply: ChanSender<Option<String>>,
}

/// The host-side `chat-mcp` IPC command the sandboxed `mcp-bridge` calls.
///
/// Each call carries one JSON-RPC line; the relay task it forwards to owns
/// the unix connection, so the MCP session state (`initialized`, in-flight
/// call ordering) survives across calls.
struct McpRelay {
    tx: ChanSender<RelayRequest>,
}

#[derive(Deserialize)]
struct McpRelayArgs {
    line: String,
}

impl IpcCommand for McpRelay {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(CHAT_MCP_IPC)
    }

    type Args = McpRelayArgs;
    type Response = Option<String>;

    async fn handle(&self, args: McpRelayArgs) -> Option<String> {
        let (reply, rx) = async_channel::bounded(1);
        self.tx
            .send(RelayRequest {
                line: args.line,
                reply,
            })
            .await
            .ok()?;
        rx.recv().await.ok().flatten()
    }
}

/// The split halves of the chat MCP connection, kept across relay calls so
/// the buffered reader never loses bytes it read ahead of a response's
/// newline. Boxed because the stream type differs per endpoint kind.
type SocketConn = (
    BufReader<Pin<Box<dyn futures_lite::AsyncRead + Send>>>,
    Pin<Box<dyn futures_lite::AsyncWrite + Send>>,
);

/// Connect one stream to the chat's MCP endpoint.
async fn connect_endpoint(endpoint: &ChatEndpoint) -> std::io::Result<SocketConn> {
    match endpoint {
        #[cfg(unix)]
        ChatEndpoint::Unix(path) => {
            let stream = async_net::unix::UnixStream::connect(path).await?;
            let (reader, writer) = futures_lite::io::split(stream);
            Ok((BufReader::new(Box::pin(reader)), Box::pin(writer)))
        }
        ChatEndpoint::Tcp(addr) => {
            let stream = async_net::TcpStream::connect(addr).await?;
            let (reader, writer) = futures_lite::io::split(stream);
            Ok((BufReader::new(Box::pin(reader)), Box::pin(writer)))
        }
    }
}

/// Spawn the relay task that owns this chat's MCP socket connection for
/// IPC-bridged calls.
fn spawn_relay(endpoint: ChatEndpoint) -> ChanSender<RelayRequest> {
    let (tx, rx) = async_channel::unbounded::<RelayRequest>();
    executor_core::spawn(async move {
        // The connection persists for the sandbox's lifetime: MCP session
        // state is per-connection, so a fresh connect per line would drop
        // `initialize` before every `tools/call`.
        let mut conn: Option<SocketConn> = None;
        let mut buf = String::new();
        while let Ok(request) = rx.recv().await {
            if conn.is_none() {
                conn = match connect_endpoint(&endpoint).await {
                    Ok(conn) => Some(conn),
                    Err(error) => {
                        warn!(%error, ?endpoint, "chat-mcp relay connect failed");
                        let _ = request.reply.send(None).await;
                        continue;
                    }
                };
            }
            let (reader, writer) = conn.as_mut().expect("connect above sets it");
            match relay_line(reader, writer, &request.line, &mut buf).await {
                Ok(response) => {
                    let _ = request.reply.send(response).await;
                }
                Err(error) => {
                    warn!(%error, "chat-mcp relay exchange failed; reconnecting");
                    conn = None;
                    let _ = request.reply.send(None).await;
                }
            }
        }
    })
    .detach();
    tx
}

/// Write one JSON-RPC line to the socket and, for requests, read one
/// response line back. Notifications (no `id`) return `None`.
async fn relay_line<R, W>(
    reader: &mut R,
    writer: &mut W,
    line: &str,
    buf: &mut String,
) -> std::io::Result<Option<String>>
where
    R: futures_lite::AsyncBufRead + Unpin,
    W: futures_lite::AsyncWrite + Unpin,
{
    let expects_response = serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .is_some_and(|message| message.get("id").is_some());

    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    if !expects_response {
        return Ok(None);
    }
    buf.clear();
    if reader.read_line(buf).await? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "chat MCP socket closed",
        ));
    }
    Ok(Some(buf.trim_end().to_owned()))
}

impl From<GrantAccess> for Access {
    fn from(access: GrantAccess) -> Self {
        match access {
            GrantAccess::R => Self::READ,
            GrantAccess::Rw => Self::WRITE,
            GrantAccess::Rx => Self::EXEC,
            GrantAccess::Rwx => Self::WRITE | Self::EXEC,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `docker run` must mount the chat dir at its own path, carry the agent
    /// state mounts, keep the image last, and leave the agent command to be
    /// appended by `spawn_agent`.
    #[test]
    fn docker_run_args_shape() {
        let dir = std::env::temp_dir().join(format!("acpbot-args-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::fs::canonicalize(&dir).unwrap();

        let docker = DockerIsolation {
            image: "agent:dev".to_string(),
            home: "/root".to_string(),
            bridge_command: vec!["acpbot".to_string(), "mcp-bridge".to_string()],
            args: vec!["--memory=2g".to_string()],
        };
        let sticker_dir = cwd.join("stickers");
        std::fs::create_dir_all(&sticker_dir).unwrap();
        let runtime = DockerRuntime::create(&docker, &cwd, &sticker_dir).unwrap();
        let args = &runtime.run_args;

        let cwd = cwd.to_string_lossy();
        assert!(
            args.windows(2)
                .any(|w| w == ["-v", &*format!("{cwd}:{cwd}")])
        );
        assert!(args.windows(2).any(|w| w == ["-w", &*cwd]));
        assert!(args.contains(&"--memory=2g".to_string()));
        assert_eq!(args.last().unwrap(), "agent:dev");
        assert!(
            args.windows(2)
                .any(|w| w[0] == "-v" && w[1].ends_with(":/root/.devin"))
        );
        // The sticker pack is mounted read-write at its host path so the
        // agent can extend it.
        let sticker = sticker_dir.to_string_lossy().into_owned();
        assert!(
            args.windows(2)
                .any(|w| w == ["-v", &*format!("{sticker}:{sticker}")])
        );
    }

    #[test]
    fn bridge_target_args() {
        assert_eq!(BridgeTarget::Ipc.arg(), "ipc:chat-mcp");
        assert_eq!(
            BridgeTarget::DockerTcp(1234).arg(),
            "tcp:host.docker.internal:1234"
        );
        assert_eq!(
            BridgeTarget::Tcp("127.0.0.1:1234".parse().unwrap()).arg(),
            "tcp:127.0.0.1:1234"
        );
        #[cfg(unix)]
        assert_eq!(
            BridgeTarget::Unix(PathBuf::from("/tmp/x.sock")).arg(),
            "/tmp/x.sock"
        );
    }
}
