//! The daemon side of the chat-tools MCP server.
//!
//! Each chat owns an endpoint the agent's `mcp-bridge` connects to: a unix
//! socket for host-side runtimes (`none`, and `native` via the IPC relay),
//! or a loopback TCP listener for `docker` (reached from the container as
//! `host.docker.internal`). Every accepted connection is served by an
//! `aither_mcp::McpServer` whose tools are bound to that chat, so a tool
//! call can never cross conversations.

use std::path::PathBuf;

use aither_core::llm::tool::Tools;
use aither_mcp::McpServer;
use aither_mcp::transport::StreamTransport;
use futures_lite::io::BufReader;
use tracing::{info, warn};

use crate::error::McpServerError;

/// The endpoint a chat's MCP server is reachable at.
///
/// Unix platforms use a socket path for host-side runtimes; Windows has no
/// unix sockets, so every runtime there shares the loopback-TCP shape docker
/// already uses.
#[derive(Debug, Clone)]
pub enum ChatEndpoint {
    /// A unix socket path (`bare`/`native` on unix).
    #[cfg(unix)]
    Unix(PathBuf),
    /// A bound TCP address (`docker` everywhere, every runtime on Windows).
    Tcp(std::net::SocketAddr),
}

/// Bind `socket` and serve an MCP endpoint per connection.
///
/// `make_tools` runs once per connection — each session's bridge gets an
/// independent server task.
#[cfg(unix)]
pub fn spawn_unix_listener(
    socket: PathBuf,
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<(), McpServerError> {
    let listener =
        async_net::unix::UnixListener::bind(&socket).map_err(|source| McpServerError::Bind {
            path: socket.clone(),
            source,
        })?;
    info!(socket = %socket.display(), "chat tools listening");

    executor_core::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => serve(stream, &make_tools),
                Err(error) => warn!(%error, "unix listener accept failed"),
            }
        }
    })
    .detach();

    Ok(())
}

/// Bind a TCP listener and return its bound address.
///
/// `expose` selects the bind interface: docker-mode bridges connect through
/// `host.docker.internal`, which lands on the host's *external* interface
/// through the Docker VM gateway — loopback would be unreachable — so the
/// listener binds all interfaces on an ephemeral port. Host-side consumers
/// (Windows runtimes) pass `false` and stay on loopback. The port is never
/// published and only the chat-tools surface is served.
async fn spawn_tcp_listener_on(
    expose: bool,
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<std::net::SocketAddr, McpServerError> {
    let bind = if expose { "0.0.0.0:0" } else { "127.0.0.1:0" };
    let listener = async_net::TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;
    info!(%addr, "chat tools listening (tcp)");

    executor_core::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => serve(stream, &make_tools),
                Err(error) => warn!(%error, "tcp listener accept failed"),
            }
        }
    })
    .detach();

    Ok(addr)
}

/// Bind a chat's MCP endpoint for host-side runtimes: a unix socket on unix,
/// a loopback TCP port on Windows. Returns the endpoint the bridge and the
/// native relay use to reach it.
#[cfg(unix)]
pub async fn bind_host_endpoint(
    socket: PathBuf,
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<ChatEndpoint, McpServerError> {
    spawn_unix_listener(socket.clone(), make_tools)?;
    Ok(ChatEndpoint::Unix(socket))
}

/// Bind a chat's MCP endpoint for host-side runtimes: a unix socket on unix,
/// a loopback TCP port on Windows. Returns the endpoint the bridge and the
/// native relay use to reach it.
#[cfg(windows)]
pub async fn bind_host_endpoint(
    _socket: PathBuf,
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<ChatEndpoint, McpServerError> {
    spawn_tcp_listener_on(false, make_tools)
        .await
        .map(ChatEndpoint::Tcp)
}

/// Bind the docker-mode MCP endpoint (`host.docker.internal` reachable).
pub async fn bind_docker_endpoint(
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<ChatEndpoint, McpServerError> {
    spawn_tcp_listener_on(true, make_tools)
        .await
        .map(ChatEndpoint::Tcp)
}

/// Serve one accepted stream as a chat MCP session.
fn serve<S>(stream: S, make_tools: &dyn Fn() -> Tools)
where
    S: futures_lite::AsyncRead + futures_lite::AsyncWrite + Send + Unpin + 'static,
{
    let (reader, writer) = futures_lite::io::split(stream);
    let transport = StreamTransport::new(BufReader::new(reader), writer);
    let tools = make_tools();
    executor_core::spawn(async move {
        let mut server = McpServer::new(transport, tools, "chat", env!("CARGO_PKG_VERSION"));
        if let Err(error) = server.run().await {
            warn!(%error, "chat MCP connection ended with error");
        }
    })
    .detach();
}
