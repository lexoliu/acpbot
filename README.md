# acpbot

A chat bot driven by an ACP agent. `acpbot run` starts the daemon: a
botkit platform adapter (Telegram or the `botkit-cli` JSONL backend) turns
every chat event into an ACP `session/prompt` to a per-chat agent process
(`devin acp` by default — any ACP harness works), and the agent speaks
back through `chat` MCP tools served by the same binary.

## How it works

- Every inbound chat event is forwarded as an ACP prompt to an agent
  process scoped to that chat; the agent's only way to speak is through
  chat tools (`send_message`, `reply`, `send_file`, `send_sticker`,
  `react`, `edit_message`, `delete_message`, `pin_message`,
  `message_status`, `history`, `search_history`, sticker-set tools,
  `restart`).
- Telegram support covers native stickers (including importing foreign
  sticker sets), reactions, edits, deletes, pins, inline keyboards, forum
  topics, and group attention classification (`direct` vs `ambient`).
- Media flows both ways: inbound files download into `inbox/` inside the
  chat's working directory; outbound `send_file` maps MIME types to the
  right Telegram media kind.
- The agent can evolve itself: it owns everything below the managed block
  of its `AGENTS.md`, can add `.devin/skills/`, can extend its sticker
  pack, and can `restart` to load changes.
- Telegram pushes no deletion event, so deletions are learned after the
  fact: the `message_status` tool probes a message's existence and
  `history` marks records `"deleted": true` once known.
- Isolation runtimes: `native` (heel), `docker`, or `bare`.

## Quick start

```sh
cargo install acpbot
cp acpbot.example.toml acpbot.toml   # fill in platform + agent command
acpbot run --config acpbot.toml
```

See `acpbot.example.toml` for the full configuration surface: `[platform]`
selects `telegram` (token/token_env) or `cli` (unix socket or stdio), and
`[agent]` sets the harness command, model, isolation, and watchdog
intervals.

## License

MIT OR Apache-2.0, at your option.
