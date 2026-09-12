//! `acpbot mcp-bridge <target>` — the link between an agent's stdio MCP
//! transport and the daemon's per-chat MCP server.
//!
//! The agent spawns this per session via `.devin/mcp_config.json`. The
//! target selects the transport:
//!
//! - `<path>` — a unix socket (isolation `none`): a byte-level pipe.
//! - `tcp:<host>:<port>` — a TCP socket (isolation `docker`): the same pipe
//!   over the daemon's loopback listener.
//! - `ipc:<command>` — inside a `heel` sandbox (isolation `native`): each
//!   JSON-RPC line is relayed through the sandbox's IPC endpoint, which the
//!   daemon's `chat-mcp` command forwards to the chat's unix socket.
//!
//! It knows nothing about MCP beyond line framing: the daemon on the other
//! end runs the real server, so a bridge can never outlive its purpose or
//! talk to the wrong chat.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;

use serde::Serialize;

use crate::error::BridgeError;

/// Pump stdio↔`target` until either side closes.
/// Pump stdio↔`target` until either side closes.
///
/// # Errors
/// [`BridgeError`] when the target cannot be reached or the pump fails.
pub fn run(target: &str) -> Result<(), BridgeError> {
    if let Some(name) = target.strip_prefix("ipc:") {
        run_ipc(name)
    } else if let Some(addr) = target.strip_prefix("tcp:") {
        run_tcp(addr)
    } else {
        run_unix(Path::new(target))
    }
}

/// A bidirectional byte stream that can split itself for the two pump
/// directions and half-close its write side when stdin ends.
trait Duplex: Read + Write + Send + 'static {
    /// A second handle writing to the same stream.
    fn dup(&self) -> std::io::Result<Self>
    where
        Self: Sized;
    /// Signal EOF to the peer without closing the read side.
    fn close_write(&self) -> std::io::Result<()>;
}

impl Duplex for UnixStream {
    fn dup(&self) -> std::io::Result<Self> {
        self.try_clone()
    }
    fn close_write(&self) -> std::io::Result<()> {
        self.shutdown(std::net::Shutdown::Write)
    }
}

impl Duplex for std::net::TcpStream {
    fn dup(&self) -> std::io::Result<Self> {
        self.try_clone()
    }
    fn close_write(&self) -> std::io::Result<()> {
        self.shutdown(std::net::Shutdown::Write)
    }
}

fn run_unix(socket: &Path) -> Result<(), BridgeError> {
    pump(UnixStream::connect(socket)?)
}

fn run_tcp(addr: &str) -> Result<(), BridgeError> {
    pump(std::net::TcpStream::connect(addr)?)
}

/// stdin→stream and stream→stdout until either side closes.
fn pump(stream: impl Duplex) -> Result<(), BridgeError> {
    let mut writer = stream.dup()?;
    let mut reader = stream;

    // stdin → stream. Runs on its own thread so it can block.
    let inbound = std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut buf = [0u8; 8192];
        loop {
            match stdin.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if writer.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
            }
        }
        // Our stdin is done — half-close so the daemon sees EOF if the
        // stream is the only writer left.
        let _ = writer.close_write();
    });

    // stream → stdout.
    {
        let mut stdout = std::io::stdout().lock();
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdout.write_all(&buf[..n]).is_err() {
                        break;
                    }
                    let _ = stdout.flush();
                }
            }
        }
    }

    let _ = inbound.join();
    Ok(())
}

/// Arguments sent to the daemon's `chat-mcp` IPC command.
#[derive(Serialize)]
struct IpcRelayArgs<'a> {
    line: &'a str,
}

/// Relay JSON-RPC lines through the sandbox's IPC endpoint.
///
/// One `chat-mcp` call per stdin line; the daemon answers with the server's
/// response line (or `None` for notifications, which produce no output).
fn run_ipc(command: &str) -> Result<(), BridgeError> {
    let endpoint = std::env::var_os("HEEL_IPC_ENDPOINT").ok_or(BridgeError::MissingIpcEndpoint)?;
    let mut client = heel::IpcClient::connect(&endpoint).map_err(BridgeError::IpcConnect)?;

    let stdin = BufReader::new(std::io::stdin().lock());
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response: Option<String> = client
            .call(command, &IpcRelayArgs { line: &line })
            .map_err(|e| BridgeError::IpcCall(command.to_string(), e))?;
        if let Some(response) = response {
            stdout.write_all(response.as_bytes())?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
    Ok(())
}
