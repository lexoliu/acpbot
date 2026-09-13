//! acpbot — a chat bot driven by an ACP agent.
//!
//! `acpbot run` starts the daemon: botkit platform adapters turn every chat
//! event into an ACP `session/prompt` to a per-chat agent process (default
//! `devin acp`), and the agent speaks back through `chat` MCP tools served by
//! this same binary (`acpbot mcp-bridge` is the stdio↔socket link the agent
//! spawns).

mod agent;
mod bot;
mod bridge;
mod chat;
mod config;
mod error;
mod handler;
mod history;
mod mcpserver;
mod sandbox;
mod sender;
mod stickerlib;
mod stickers;
mod stickerset;
mod tools;

use std::path::{Path, PathBuf};

use botkit_core::{Bot, Shutdown};
use botkit_telegram::TelegramClient;
use tracing::{info, warn};

use crate::agent::Dispatcher;
use crate::config::{Config, PlatformConfig};
use crate::error::MainError;
use crate::sender::Sender;
use crate::stickerset::StickerSet;

fn main() -> Result<(), MainError> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "acpbot=info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("mcp-bridge") => {
            let Some(target) = args.get(1) else {
                return Err(MainError::Usage(
                    "usage: acpbot mcp-bridge <socket|tcp:HOST:PORT|ipc:COMMAND>".to_string(),
                ));
            };
            bridge::run(target).map_err(MainError::from)
        }
        Some("run") => {
            let config = match args.as_slice() {
                [_, flag, path] if flag == "--config" => PathBuf::from(path),
                [_] => PathBuf::from("acpbot.toml"),
                _ => {
                    return Err(MainError::Usage(
                        "usage: acpbot run [--config PATH]".to_string(),
                    ));
                }
            };
            run(&config)
        }
        _ => Err(MainError::Usage(
            "usage: acpbot run [--config PATH] | acpbot mcp-bridge              <socket|tcp:HOST:PORT|ipc:COMMAND>"
                .to_string(),
        )),
    }
}

fn run(config_path: &Path) -> Result<(), MainError> {
    let mut config = Config::load(config_path)?;
    info!(config = %config_path.display(), data_dir = %config.paths.data_dir.display(), "starting acpbot");

    std::fs::create_dir_all(config.paths.data_dir.join("chats"))?;
    std::fs::create_dir_all(config.paths.data_dir.join("run"))?;

    // Everything derived from `data_dir` — the session cwd, the bridge
    // socket arg, the sticker path inside AGENTS.md — is resolved against
    // the agent's own working directory, not the daemon's: a relative
    // `data_dir` would point each of those at a path that doesn't exist.
    config.paths.data_dir = std::fs::canonicalize(&config.paths.data_dir)?;

    let sticker_dir = config.paths.sticker_dir();
    std::fs::create_dir_all(&sticker_dir)?;

    // botkit spawns handler work through executor-core; give it a global
    // executor before anything registers.
    let executor: &'static async_executor::Executor<'static> =
        Box::leak(Box::new(async_executor::Executor::new()));
    executor_core::init_global_executor(executor);

    info!("agent: {} {:?}", config.agent.command, config.agent.args);
    let (events_tx, events_rx) = async_channel::unbounded();

    // Platform-specific pieces: the bot driving events in, and the factory
    // binding each chat's tool sender. Everything downstream of them is
    // platform-agnostic.
    type BotRun = std::pin::Pin<Box<dyn Future<Output = Result<(), botkit_core::BotError>> + Send>>;
    let (signal, shutdown) = Shutdown::channel();
    let (sender_for, bot_run, telegram_client): (
        crate::agent::SenderFactory,
        BotRun,
        Option<TelegramClient>,
    ) = match &config.platform {
        PlatformConfig::Telegram { .. } => {
            let token = config.platform.telegram_token()?;
            let client = TelegramClient::new(token.clone());
            let (set_name, set_title, set_owner) = config.platform.telegram_sticker_set();
            let stickers = std::sync::Arc::new(StickerSet::new(
                client.clone(),
                sticker_dir.clone(),
                set_name,
                set_title,
                set_owner,
            ));
            // Group attention classification needs the bot's own identity;
            // `can_read_all_group_messages` also tells us whether ambient
            // group traffic can reach us at all.
            let me = futures_lite::future::block_on(client.get_me())?;
            if me.can_read_all_group_messages == Some(false) {
                warn!(
                    "bot privacy mode is ON — Telegram only delivers commands, \
                     replies and mentions; ambient group messages never arrive. \
                     Disable it via @BotFather for topic-following"
                );
            }
            info!(bot = ?me.username, id = me.id, "bot identity");
            let me = bot::BotIdentity {
                id: me.id,
                username: me.username.clone(),
            };
            info!("sticker pack scans on demand");
            let bot = bot::build(token, events_tx.clone(), me);
            let sender_client = client.clone();
            (
                Box::new(move |key, spoke, thread, last_action| {
                    Sender::telegram(
                        sender_client.clone(),
                        key,
                        spoke,
                        thread,
                        last_action,
                        stickers.clone(),
                    )
                }),
                Box::pin(bot.run_until(shutdown)),
                Some(client),
            )
        }
        PlatformConfig::Cli { .. } => {
            let transport = config.platform.cli_transport()?;
            let cli_bot = bot::build_cli(transport, events_tx.clone());
            let hub = cli_bot.hub();
            (
                Box::new(move |key, spoke, thread, last_action| {
                    Ok(Sender::cli(hub.clone(), key, spoke, thread, last_action))
                }),
                Box::pin(cli_bot.run_until(shutdown)),
                None,
            )
        }
        PlatformConfig::Discord { .. } => {
            let token = config.platform.discord_token()?;
            let application_id = config.platform.discord_application_id()?;
            let client = botkit_discord::DiscordClient::new(token.clone(), application_id.clone());
            // Group attention classification needs the bot's own identity:
            // guild messages mentioning or replying to it are `direct`.
            let me = futures_lite::future::block_on(client.current_user())?;
            info!(bot = ?me.username, id = %me.id, "bot identity");
            let bot = bot::build_discord(
                token,
                application_id,
                events_tx.clone(),
                bot::DiscordIdentity { id: me.id },
            );
            (
                Box::new(move |key, spoke, thread, last_action| {
                    Ok(Sender::discord(
                        client.clone(),
                        key,
                        spoke,
                        thread,
                        last_action,
                    ))
                }),
                Box::pin(bot.run_until(shutdown)),
                None,
            )
        }
    };

    let sticker_library = std::sync::Arc::new(stickerlib::StickerLibrary::load(
        &config.paths.data_dir,
        telegram_client,
    )?);

    let dispatcher = Dispatcher::new(
        config.agent,
        std::env::current_exe()?,
        config.paths.data_dir.clone(),
        sticker_dir,
        config.persona.clone(),
        sender_for,
        sticker_library,
    );
    // Kept, not detached: on shutdown the dispatcher's drop chain is what
    // kills every chat's agent process (a heel sandbox kills on drop).
    let dispatcher_task = executor_core::spawn(dispatcher.run(events_rx));

    ctrlc::set_handler(move || signal.shutdown()).expect("install Ctrl-C handler");

    info!("polling for events");
    futures_lite::future::block_on(executor.run(bot_run)).map_err(MainError::Bot)?;

    // The bot is stopped; closing the last event sender ends the dispatcher,
    // which waits for every chat actor (and its sandbox) to finish.
    drop(events_tx);
    info!("shutting down agents");
    futures_lite::future::block_on(executor.run(dispatcher_task));
    Ok(())
}
