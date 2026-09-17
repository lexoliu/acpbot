# acpbot

A chat bot driven by an ACP agent. `acpbot run` starts the daemon: a botkit
platform adapter (Telegram, Discord, or the `botkit-cli` JSONL backend) turns
every chat
event into an ACP `session/prompt` to one shared agent process (default
`devin acp`) — every conversation feeds the same context window — and the
agent speaks back through `chat` MCP tools served by the same binary.
`acpbot mcp-bridge` is the stdio↔socket link the agent spawns.

## Build, test, run

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
./target/debug/acpbot run --config acpbot.toml
```

`executor_core::spawn` requires a global executor; `main` installs a leaked
`async_executor::Executor` before anything registers. Tests that spawn tasks
must do the same (see `ensure_executor` in test modules).

## Architecture

- `bot.rs` — platform wiring: every update becomes a `ChatEvent` on the
  dispatcher channel (`build`/`build_discord`/`build_cli`). Group messages get
  `attention`: `"direct"` (reply to the bot, @-mention, command, button) vs
  `"ambient"` (room chatter the agent may still answer). Needs the bot's own
  identity — `getMe`/`GET /users/@me` — fetched once at startup.
- `chat.rs` — `ChatEvent`, the JSON schema the agent sees; it is documented
  again in the agent's `AGENTS.md` managed block, so the two never drift.
- `agent.rs` — `Dispatcher` + the single shared `ChatActor` + `ChatRouter`:
  spawns the ACP process, restores the session (`session/resume` →
  `session/load` → `session/new`), forwards event batches as prompts, runs
  the first-reply watchdog and the idle-compaction timer. The router maps
  `ChatKey` → per-chat `Sender`/`History`/topic cell and tracks the
  turn's triggering chat, which tools' optional `chat` argument defaults to.
- `sender.rs` — `Platform` enum (`Telegram`, `Discord`, `Cli`, test `Record`):
  every
  outbound tool call lands here. `probe_message` checks whether a message
  still exists (a no-op `editMessageReplyMarkup` answers "not modified" on
  a live message, "to edit not found" on a deleted one) — Telegram pushes
  no deletion update, so deletions are only ever learned after the fact:
  via the `message_status` tool or `"deleted": true` marks on `history`
  records, never as an inbound event.
- `mcpserver.rs`, `bridge.rs`, `sandbox.rs` — the chat-tools MCP endpoint and
  the isolation runtimes (`native` heel, `docker`, `bare`).
- `history.rs` — `history.jsonl` per chat: inbound events appended by the
  actor, outbound actions by `Sender`, read back by the `history` and
  `search_history` tools (epoch/RFC3339/`30m`-style `since`/`until`).
- `stickers.rs`, `stickerset.rs` — the agent-evolvable sticker pack and its
  Telegram sticker-set publishing.
- `stickerlib.rs` — the persistent catalog of foreign sticker sets
  (`sticker_library.json` under `data_dir`): a bot may send any hosted
  sticker by `file_id`, so importing a set only needs `getStickerSet` once.

## Conventions

- Public trait methods return `impl Future`; never expose `Pin<Box<dyn Future>>`
  or boxed-future aliases in public signatures.
- When a trait must be stored dynamically, define a private object-safe twin
  `[Trait]Impl` plus a public wrapper `Any[Trait]` (e.g. `AnyChatActionSender`).
  Boxing lives inside the wrapper — callers only see the concrete API.
- `anyhow` is banned: every error is a `thiserror` enum in `src/error.rs`, one
  per subsystem (`ConfigError`, `SenderError`, `AgentError`, `SandboxError`,
  `BridgeError`, `McpServerError`, `StickerSetError`, `StickersError`,
  `MainError`), so callers can match on failure kinds.
- Fail fast: a `warn!` + `continue`/`None` is only for genuinely optional
  work (media downloads, capability probes); everything else propagates.
- `tracing` for diagnostics, never `println!` (except `mcp-bridge`, which is a
  byte-level stdio pipe by design).

## Config

See `acpbot.example.toml`. `[platform]` selects `telegram` (token/token_env),
`discord` (token/token_env + `application_id`), or `cli` (`socket` unix path
or `stdio = true`). `[agent]` sets the harness
command, model, isolation, and `idle_compact_secs`.
