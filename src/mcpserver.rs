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

/// Bind `socket` and serve an MCP endpoint per connection.
///
/// `make_tools` runs once per connection — each session's bridge gets an
/// independent server task.
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

/// Bind a TCP listener for docker-mode bridges.
///
/// Containers cannot connect to a host unix socket, so their `mcp-bridge`
/// reaches the daemon on `host.docker.internal:<port>` instead. That name
/// lands on the host's *external* interface through the Docker VM gateway —
/// loopback would be unreachable — so the listener binds all interfaces on
/// an ephemeral port. The port is never published and only the chat-tools
/// surface is served.
pub async fn spawn_tcp_listener(
    make_tools: impl Fn() -> Tools + Send + 'static,
) -> Result<u16, McpServerError> {
    let listener = async_net::TcpListener::bind("0.0.0.0:0").await?;
    let port = listener.local_addr()?.port();
    info!(port, "chat tools listening (tcp)");

    executor_core::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => serve(stream, &make_tools),
                Err(error) => warn!(%error, "tcp listener accept failed"),
            }
        }
    })
    .detach();

    Ok(port)
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
