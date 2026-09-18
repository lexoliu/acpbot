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
  spawns the ACP process at daemon start (process, handshake, `session/new`,
  and the continuity inject are all paid before the first event arrives),
  always opens a *fresh* session and injects the previous incarnation's
  `CONTINUITY.md` as the bootstrap prompt; before a clean close it asks
  the agent to (re)write that file.
  A session id in `sessions.json` is a "died before handoff" marker — the
  next spawn restores it only to extract the summary, never to resume.
  Forwards event batches as prompts, runs the first-reply watchdog and
  the idle-compaction timer. Every event pulled off the wire is journaled
  to `inflight.json` — `sent` while its turn is unconfirmed, `queued`
  when coalesced mid-turn — and a confirmed turn end clears it, so a
  crash or shutdown replays exactly what the agent never finished
  (events already in IM history are re-prompted, never re-logged; the
  prompt flags replays so the model checks `history` before
  re-answering). The router maps
  `ChatKey` → per-chat `Sender`/`History`/topic cell and tracks the
  turn's triggering chat, which tools' optional `chat` argument defaults to.
- `sender.rs` — `Platform` enum (`Telegram`, `Discord`, `Cli`, test `Record`):
  every
  outbound tool call lands here. Text first passes `unescape_escapes`,
  which rewrites the JSON escapes the model sometimes emits literally
  (`\n`, `\t`, `\uXXXX`, …) into real characters while skipping code
  spans/fences and preserving markdown escapes like `\*`. Telegram-bound
  text (and media captions)
  is rendered from markdown into `entities` via
  `botkit_telegram::markdown::render` — offsets computed here, so emphasis
  next to CJK or full-width punctuation works and no `parse_mode` escaping
  is involved. `probe_message` checks whether a message
  still exists (a no-op `editMessageReplyMarkup` answers "not modified" on
  a live message, "to edit not found" on a deleted one) — Telegram pushes
  no deletion update, so deletions are only ever learned after the fact:
  via the `message_status` tool or `"deleted": true` marks on `history`
  records, never as an inbound event.
- `mcpserver.rs`, `bridge.rs`, `sandbox.rs` — the chat-tools MCP endpoint and
  the isolation runtimes (`native` heel, `docker`, `bare`).
- `agy.rs` — `acpbot agy-bridge`: an ACP server over stdio that fronts the
  Antigravity `agy` CLI (`[agent] command = "<acpbot>", args = ["agy-bridge"]`).
  agy runs one `stream-json` process per turn; the bridge threads its
  `conversation_id` via `--conversation` and persists the `acp session id →
  conversation id` map at `<cwd>/.agy-bridge.json` so `session/load` can
  still recover a session for the continuity handoff. Chat tools reach agy
  through its global `~/.gemini/config/mcp_config.json`, mirrored from the
  session's `.devin/mcp_config.json`. Requires `isolation.kind = "none"`.
- `history.rs` — `history.jsonl` per chat: the durable IM record, not an
  agent transcript — inbound events appended by the actor, outbound actions
  by `Sender`, read back by the `history` and `search_history` tools
  (epoch/RFC3339/`30m`-style `since`/`until`). It outlives sessions by
  design; the ACP session context is disposable.
- `registry.rs` — `chats.json` at the data-dir root: every chat the bot
  has state for (`platform:id` → type/title/activity times/forum topics),
  noted on each inbound event and on an outbound first touch, read back
  by `list_chats`. No platform offers a chat-enumeration API, so the bot
  keeps its own.
- `stickers.rs`, `stickerset.rs` — the agent-evolvable sticker pack and its
  Telegram sticker-set publishing.
- `stickerlib.rs` — the persistent catalog of foreign sticker sets
  (`sticker_library.json` under `data_dir`): a bot may send any hosted
  sticker by `file_id`, so importing a set only needs `getStickerSet` once.
- `browser.rs`, `cdp.rs` — the daemon-hosted browser (`[browser]`, on by
  default): one real Chrome launched lazily and driven over a hand-rolled
  CDP client, shared by the main session and subagents through `browser_*`
  tools on the same `chat` MCP endpoint. Stealth by construction — no
  `--enable-automation`, and the client never sends `Runtime.enable`;
  profile persists at `<data_dir>/browser-profile`.
- `preview.rs` — link previews on inbound text (`[preview]`, on by
  default): each `http(s)` link in a message is fetched at prompt-assembly
  time (like `media.file`, so journal replays regenerate it) and attached
  as `link_previews` — `og:` title/site/description, or the post body for
  `t.me/<channel>/<post>` links (fetched via the server-rendered `/s/`
  embed). Per-link timeout + concurrency, a short TTL cache, and a
  loopback/private-host short-circuit since fetches run on the host
  network.

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

## Tuning the agent

The one shared agent is this bot's UI main thread: while its turn runs,
every chat waits. Keep the main turn short — ack, hand the task to a
background subagent, end the turn; relay the result to the asking chat when
the subagent reports back. Tooling, prompts, and the managed `AGENTS.md`
block (`PROTOCOL_DOC` in `agent.rs`) all tune toward that: work happens in
subagents, the main turn only talks and dispatches. It is also enforced —
`BotClientHandler` flags any tool call with `kind: Execute` the moment one
starts on the shared thread — the agy bridge classifies `run_command` &
friends, devin's `exec` reports it natively, and a `MAIN_BLOCKED_TOOLS`
name list covers harnesses that leave `kind` unset. (Subagent calls run in
their own conversations and never surface here.) The turn is cancelled and
re-prompted as a `nudge`, bounded by the same retry budget as the
silent-turn recovery.

## Config

See `acpbot.example.toml`. `[platform]` selects `telegram` (token/token_env),
`discord` (token/token_env + `application_id`), or `cli` (`socket` unix path
or `stdio = true`). `[agent]` sets the harness
command, model, isolation, and `idle_compact_secs`.
