//! The daemon-hosted browser — one real Chrome, driven over CDP by the
//! daemon itself and shared by the agent and every subagent.
//!
//! ## Why hand-rolled CDP
//!
//! Anti-bot walls (Cloudflare, DataDome) fingerprint automation on two
//! channels: launch flags (`--enable-automation` flips
//! `navigator.webdriver`, and headless-driver arg sets read like a
//! bot's), and CDP chatter — `Runtime.enable` in particular opens a
//! side channel page scripts can probe (the rebrowser trick every
//! "stealth" Playwright fork exists to dodge). Driver crates send it
//! unconditionally on every frame attach. This client only ever sends
//! what `cdp.rs` writes: `Page`, `DOM`, `Input`, `Accessibility`,
//! `Log`, `Network`, `Target` — never `Runtime.enable`, and no injected
//! shim scripts either since a clean launch needs none. Input goes
//! through `Input.dispatch*`, which produces real trusted events.
//!
//! ## Lifecycle
//!
//! [`Browser`] is a cheap handle; Chrome launches lazily on the first
//! `browser_*` call and stays up until it crashes (relaunched on the
//! next call) or the daemon drops it. The profile lives under
//! `data_dir/browser-profile`, so cookies and logins survive restarts.
//! A `Mutex` serializes the one browser across all callers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use aither_core::llm::tool::{Tool, ToolResult, ToolResultPart, Tools};
use async_lock::Mutex;
use base64::Engine as _;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::cdp::{Cdp, devtools_ws_url};
use crate::config::BrowserConfig;
use crate::error::BrowserError;

/// One shared browser handle — cloneable, cheap, launches Chrome lazily.
#[derive(Clone)]
pub struct Browser {
    inner: Arc<Mutex<State>>,
    config: BrowserConfig,
    /// `data_dir` — the profile and artifact dirs live under it.
    data_dir: PathBuf,
}

/// Down until first use; `Up` holds the live browser (boxed — the enum
/// stays small next to `Down`).
enum State {
    Down,
    Up(Box<Live>),
}

/// A running Chrome and the page we're attached to.
struct Live {
    /// The Chrome child — killed when the `Live` drops or relaunches.
    child: async_process::Child,
    /// The DevTools connection.
    cdp: Cdp,
    /// Flat session id for the current page target.
    session: String,
    /// Target id of the current page (for tab ops and liveness).
    target: String,
    /// Main frame id — needed to create the isolated world.
    frame: String,
    /// `Page.createIsolatedWorld` result, made on first `browser_evaluate`.
    world: Option<i64>,
    /// Snapshot element refs: `e1` → `backendDOMNodeId`. Cleared on
    /// navigation and re-attach — refs only make sense for the snapshot
    /// that minted them.
    refs: HashMap<u32, i64>,
    /// Next ref number.
    next_ref: u32,
    /// Where screenshots and other artifacts land.
    out_dir: PathBuf,
}

impl Drop for Live {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

impl Browser {
    /// A handle that will launch Chrome on first use.
    pub fn new(config: BrowserConfig, data_dir: PathBuf) -> Self {
        Self {
            inner: Arc::new(Mutex::new(State::Down)),
            config,
            data_dir,
        }
    }

    /// Register the `browser_*` tools into `tools`.
    pub fn register_into(&self, tools: &mut Tools) {
        macro_rules! tool {
            ($ty:ty) => {
                tools
                    .register(<$ty>::new(self.clone()))
                    .expect("browser tool registration is static")
            };
        }
        tool!(Navigate);
        tool!(Snapshot);
        tool!(Click);
        tool!(TypeText);
        tool!(PressKey);
        tool!(Hover);
        tool!(Scroll);
        tool!(Evaluate);
        tool!(Screenshot);
        tool!(WaitFor);
        tool!(Tabs);
        tool!(NavigateBack);
        tool!(ConsoleLog);
        tool!(NetworkLog);
        tool!(CloseBrowser);
    }

    /// The live browser, launching or relaunching as needed. The lock is
    /// held for the whole tool call — there is one browser, so calls
    /// serialize naturally.
    async fn live(&self) -> Result<LiveGuard<'_>, BrowserError> {
        let mut state = self.inner.lock().await;
        let healthy = match &mut *state {
            State::Up(live) => live.healthy().await,
            State::Down => false,
        };
        if !healthy {
            if matches!(*state, State::Up(_)) {
                warn!("browser died — relaunching");
            }
            *state = State::Up(Box::new(launch(&self.config, &self.data_dir).await?));
        }
        Ok(LiveGuard { state })
    }
}

/// A `MutexGuard<State>` narrowed to the `Up` variant for tool bodies.
struct LiveGuard<'a> {
    state: async_lock::MutexGuard<'a, State>,
}

impl std::ops::Deref for LiveGuard<'_> {
    type Target = Live;
    fn deref(&self) -> &Live {
        match &*self.state {
            State::Up(live) => live,
            State::Down => unreachable!("guard only exists while Up"),
        }
    }
}

impl std::ops::DerefMut for LiveGuard<'_> {
    fn deref_mut(&mut self) -> &mut Live {
        match &mut *self.state {
            State::Up(live) => live,
            State::Down => unreachable!("guard only exists while Up"),
        }
    }
}

impl Live {
    /// Socket connected, child running, and the attached page still
    /// there (a closed tab or crashed renderer re-attaches to another
    /// page or makes a fresh one).
    async fn healthy(&mut self) -> bool {
        self.cdp.is_alive()
            && matches!(self.child.try_status(), Ok(None))
            && self.ensure_target().await.is_ok()
    }

    /// Re-resolve `self.target`: still listed → keep; gone → attach to
    /// the first remaining page or create a fresh one.
    async fn ensure_target(&mut self) -> Result<(), BrowserError> {
        let targets = self.targets().await?;
        if targets.iter().any(|t| t.0 == self.target) {
            return Ok(());
        }
        let (target, ..) = match targets.into_iter().next() {
            Some(t) => t,
            None => self.new_target("about:blank").await?,
        };
        self.attach(&target).await
    }

    /// `Target.getTargets` → page targets as (id, title, url).
    async fn targets(&self) -> Result<Vec<(String, String, String)>, BrowserError> {
        let result = self.cdp.call(None, "Target.getTargets", json!({})).await?;
        Ok(result["targetInfos"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|t| t["type"].as_str() == Some("page"))
            .map(|t| {
                (
                    t["targetId"].as_str().unwrap_or_default().to_string(),
                    t["title"].as_str().unwrap_or_default().to_string(),
                    t["url"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect())
    }

    /// `Target.createTarget` → (id, title, url).
    async fn new_target(&self, url: &str) -> Result<(String, String, String), BrowserError> {
        let result = self
            .cdp
            .call(
                None,
                "Target.createTarget",
                json!({"url": url, "newWindow": false, "background": false}),
            )
            .await?;
        let id = result["targetId"]
            .as_str()
            .ok_or_else(|| BrowserError::Protocol("createTarget: no targetId".to_string()))?;
        Ok((id.to_string(), String::new(), url.to_string()))
    }

    /// Attach to a page target flat-mode and enable the domains we use.
    /// `Runtime.enable` is deliberately absent — see module docs.
    async fn attach(&mut self, target: &str) -> Result<(), BrowserError> {
        let result = self
            .cdp
            .call(
                None,
                "Target.attachToTarget",
                json!({"targetId": target, "flatten": true}),
            )
            .await?;
        self.session = result["sessionId"]
            .as_str()
            .ok_or_else(|| BrowserError::Protocol("attachToTarget: no sessionId".to_string()))?
            .to_string();
        self.target = target.to_string();
        self.refs.clear();
        self.world = None;
        for method in ["Page.enable", "Log.enable", "Network.enable"] {
            self.call(method, json!({})).await?;
        }
        let tree = self.call("Page.getFrameTree", json!({})).await?;
        self.frame = tree["frameTree"]["frame"]["id"]
            .as_str()
            .ok_or_else(|| BrowserError::Protocol("getFrameTree: no frame id".to_string()))?
            .to_string();
        Ok(())
    }

    /// A session-scoped CDP call on the current page.
    async fn call(&self, method: &str, params: Value) -> Result<Value, BrowserError> {
        self.cdp.call(Some(&self.session), method, params).await
    }

    /// `Runtime.evaluate` in the page's main world — used for reads
    /// (readyState, selector probes); it leaves nothing behind.
    async fn eval(&self, expression: &str) -> Result<Value, BrowserError> {
        let result = self
            .call(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
            )
            .await?;
        eval_result(result)
    }

    /// `Runtime.evaluate` inside our isolated world — agent scripts
    /// can't leak globals into the page's world.
    async fn eval_isolated(&mut self, expression: &str) -> Result<Value, BrowserError> {
        if self.world.is_none() {
            let created = self
                .call("Page.createIsolatedWorld", json!({"frameId": self.frame}))
                .await?;
            self.world = created["executionContextId"].as_i64();
        }
        let result = self
            .call(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "contextId": self.world,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
            )
            .await?;
        eval_result(result)
    }

    /// The backend node id behind a snapshot ref, or a nudge to
    /// re-snapshot.
    fn backend(&self, reference: u32) -> Result<i64, BrowserError> {
        self.refs.get(&reference).copied().ok_or_else(|| {
            BrowserError::Protocol(format!(
                "no element e{reference} — take a fresh browser_snapshot"
            ))
        })
    }

    /// Point at the center of an element's first quad, scrolling it
    /// into view first. Errors when the element is gone or unrendered.
    async fn point_at(&mut self, reference: u32) -> Result<(f64, f64), BrowserError> {
        let backend = self.backend(reference)?;
        self.call(
            "DOM.scrollIntoViewIfNeeded",
            json!({"backendNodeId": backend}),
        )
        .await?;
        let quads = self
            .call("DOM.getContentQuads", json!({"backendNodeId": backend}))
            .await?;
        let quad = quads["quads"]
            .as_array()
            .and_then(|q| q.first())
            .and_then(Value::as_array)
            .ok_or_else(|| {
                BrowserError::Protocol(format!("e{reference} has no box — hidden or detached"))
            })?;
        // quad = [x1,y1, x2,y2, x3,y3, x4,y4] — center of the box.
        let x = (quad[0].as_f64().unwrap_or(0.0) + quad[4].as_f64().unwrap_or(0.0)) / 2.0;
        let y = (quad[1].as_f64().unwrap_or(0.0) + quad[5].as_f64().unwrap_or(0.0)) / 2.0;
        Ok((x, y))
    }

    /// A trusted mouse click at (x, y): approach move, press, human-ish
    /// hold, release.
    async fn click_at(&self, x: f64, y: f64) -> Result<(), BrowserError> {
        let (jx, jy) = jitter();
        let (x, y) = (x + jx, y + jy);
        self.mouse("mouseMoved", x / 2.0, y / 2.0).await?;
        self.mouse("mouseMoved", x, y).await?;
        self.mouse_button("mousePressed", x, y).await?;
        async_io::Timer::after(std::time::Duration::from_millis(35 + jitter_ms())).await;
        self.mouse_button("mouseReleased", x, y).await
    }

    async fn mouse(&self, kind: &str, x: f64, y: f64) -> Result<(), BrowserError> {
        self.call(
            "Input.dispatchMouseEvent",
            json!({"type": kind, "x": x, "y": y}),
        )
        .await?;
        Ok(())
    }

    async fn mouse_button(&self, kind: &str, x: f64, y: f64) -> Result<(), BrowserError> {
        self.call(
            "Input.dispatchMouseEvent",
            json!({"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1}),
        )
        .await?;
        Ok(())
    }

    /// Wait for `document.readyState === "complete"`, bounded.
    async fn wait_loaded(&self, timeout: std::time::Duration) -> Result<(), BrowserError> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self
                .eval("document.readyState")
                .await
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .as_deref()
                == Some("complete")
            {
                return Ok(());
            }
            if std::time::Instant::now() > deadline {
                return Err(BrowserError::Timeout("page load".to_string()));
            }
            async_io::Timer::after(std::time::Duration::from_millis(200)).await;
        }
    }

    /// Current page title + url for tool replies.
    async fn location(&self) -> (String, String) {
        let title = self
            .eval("document.title")
            .await
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let url = self
            .eval("location.href")
            .await
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        (title, url)
    }
}

/// Small cursor jitter — nanos-derived, no rng dep needed for ±2px.
fn jitter() -> (f64, f64) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    ((nanos % 5) as f64 - 2.0, ((nanos / 7) % 5) as f64 - 2.0)
}

/// 0–60ms of extra hold time between press and release.
fn jitter_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from((d.subsec_nanos() / 11) as u8 % 60))
        .unwrap_or(20)
}

/// Pull a JS value out of a `Runtime.evaluate` response, surfacing
/// exceptions as errors.
fn eval_result(result: Value) -> Result<Value, BrowserError> {
    if let Some(exception) = result.get("exceptionDetails") {
        let text = exception["exception"]["description"]
            .as_str()
            .or_else(|| exception["text"].as_str())
            .unwrap_or("js exception");
        return Err(BrowserError::Protocol(text.to_string()));
    }
    let remote = &result["result"];
    Ok(match remote.get("value") {
        Some(value) => value.clone(),
        // unserializable / undefined → its description or type name.
        None => remote["description"]
            .as_str()
            .map(|d| json!(d))
            .unwrap_or_else(|| json!(remote["type"].as_str().unwrap_or("undefined"))),
    })
}

/// Launch Chrome and attach to a page.
async fn launch(config: &BrowserConfig, data_dir: &Path) -> Result<Live, BrowserError> {
    let executable = match &config.executable {
        Some(path) => path.clone(),
        None => find_chrome()?,
    };
    let profile = config
        .profile_dir
        .clone()
        .unwrap_or_else(|| data_dir.join("browser-profile"));
    let out_dir = data_dir.join("browser");
    std::fs::create_dir_all(&profile).map_err(BrowserError::Io)?;
    std::fs::create_dir_all(&out_dir).map_err(BrowserError::Io)?;

    let mut args = vec![
        "--remote-debugging-port=0".to_string(),
        format!("--user-data-dir={}", profile.display()),
        // Chrome ≥111 rejects ws handshakes carrying a foreign Origin.
        "--remote-allow-origins=*".to_string(),
        "--no-first-run".to_string(),
        "--no-default-browser-check".to_string(),
        // Kills navigator.webdriver even if something sets the flag.
        "--disable-blink-features=AutomationControlled".to_string(),
    ];
    if config.headless {
        args.push("--headless=new".to_string());
    }
    args.extend(config.args.iter().cloned());

    info!(exe = %executable.display(), headless = config.headless, "launching browser");
    let child = async_process::Command::new(&executable)
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(BrowserError::Io)?;

    let ws_url = devtools_ws_url(&profile).await?;
    let cdp = Cdp::connect(&ws_url).await?;
    let mut live = Live {
        child,
        cdp,
        session: String::new(),
        target: String::new(),
        frame: String::new(),
        world: None,
        refs: HashMap::new(),
        next_ref: 0,
        out_dir,
    };
    // Attach to the first existing page, or open one.
    let target = match live.targets().await?.into_iter().next() {
        Some((id, ..)) => id,
        None => live.new_target("about:blank").await?.0,
    };
    live.attach(&target).await?;
    info!("browser ready");
    Ok(live)
}

/// Find a Chrome-family binary: config first, then the usual installs.
fn find_chrome() -> Result<PathBuf, BrowserError> {
    #[cfg(target_os = "macos")]
    const CANDIDATES: &[&str] = &[
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ];
    #[cfg(target_os = "linux")]
    const CANDIDATES: &[&str] = &[
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/microsoft-edge",
        "/snap/bin/chromium",
    ];
    #[cfg(target_os = "windows")]
    const CANDIDATES: &[&str] = &[
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    ];
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    const CANDIDATES: &[&str] = &[];

    for path in CANDIDATES {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(BrowserError::NoExecutable)
}

use std::path::Path;

// ---------- tools ----------

/// No-arg tools still need a schema type.
#[derive(Debug, Deserialize, JsonSchema)]
struct EmptyArgs {}

/// Every tool locks the browser, ensures it's live, and maps failures
/// to error results the model can read. `|$live, $argn| $body` becomes
/// an `async fn` on the tool — the binding idents are passed through
/// the pattern so the body resolves them (macro hygiene).
macro_rules! browser_tool {
    ($name:ident, $tool:literal, $desc:literal, $args:ty, |$live:ident, $argn:ident| $body:block) => {
        struct $name {
            browser: Browser,
        }
        impl $name {
            fn new(browser: Browser) -> Self {
                Self { browser }
            }
            async fn run<'a>(
                &'a self,
                $live: &'a mut Live,
                $argn: $args,
            ) -> Result<ToolResult, BrowserError> {
                $body
            }
        }
        impl Tool for $name {
            type Arguments = $args;
            type Res = ToolResult;
            fn name(&self) -> std::borrow::Cow<'static, str> {
                $tool.into()
            }
            fn description(&self) -> std::borrow::Cow<'static, str> {
                $desc.into()
            }
            async fn call(&self, args: Self::Arguments) -> aither_core::Result<Self::Res> {
                let mut guard = match self.browser.live().await {
                    Ok(guard) => guard,
                    Err(error) => {
                        return Ok(ToolResult::error(format!("browser unavailable: {error}")));
                    }
                };
                let result = self.run(&mut guard, args).await;
                Ok(result.unwrap_or_else(|error| ToolResult::error(error.to_string())))
            }
        }
    };
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NavigateArgs {
    /// The URL to open (https://, http://, file://, about:…).
    url: String,
}

browser_tool!(
    Navigate,
    "browser_navigate",
    "Open a URL in the shared browser and wait for the page to finish \
     loading. Returns the page title and final URL (after redirects).",
    NavigateArgs,
    |live, args| {
        let result = live.call("Page.navigate", json!({"url": args.url})).await?;
        live.refs.clear();
        if let Some(text) = result["errorText"].as_str()
            && text != "net::ERR_ABORTED"
        {
            return Err(BrowserError::Protocol(format!("navigate failed: {text}")));
        }
        live.wait_loaded(std::time::Duration::from_secs(30)).await?;
        let (title, url) = live.location().await;
        Ok(ToolResult::text(format!("{title}\n{url}")))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct SnapshotArgs {
    /// Only list interactive elements (links, buttons, inputs…) — the
    /// default. `false` dumps every named accessibility node.
    interactive_only: Option<bool>,
}

browser_tool!(
    Snapshot,
    "browser_snapshot",
    "Read the page as an accessibility tree: indented lines of \
     `role \"name\"`, interactive elements tagged [eN]. Use the eN refs \
     with browser_click/browser_type/browser_hover.",
    SnapshotArgs,
    |live, args| {
        let tree = live.call("Accessibility.getFullAXTree", json!({})).await?;
        Ok(ToolResult::text(render_ax_tree(
            live,
            &tree,
            args.interactive_only.unwrap_or(true),
        )))
    }
);

/// Interactive roles get eN refs.
const INTERACTIVE_ROLES: &[&str] = &[
    "link",
    "button",
    "textbox",
    "searchbox",
    "combobox",
    "listbox",
    "checkbox",
    "radio",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "tab",
    "switch",
    "slider",
    "spinbutton",
    "treeitem",
];

/// Roles that never deserve a line of their own.
const SKIP_ROLES: &[&str] = &["none", "generic", "InlineTextBox"];

/// Render `Accessibility.getFullAXTree` into the Playwright-style tree
/// the model navigates by, minting eN refs for interactive nodes.
fn render_ax_tree(live: &mut Live, tree: &Value, interactive_only: bool) -> String {
    let Some(nodes) = tree["nodes"].as_array() else {
        return "empty page".to_string();
    };
    let by_id: HashMap<&str, &Value> = nodes
        .iter()
        .filter_map(|n| n["nodeId"].as_str().map(|id| (id, n)))
        .collect();
    let mut out = String::new();

    fn value<'a>(node: &'a Value, key: &str) -> &'a str {
        node[key]["value"].as_str().unwrap_or_default()
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        id: &str,
        depth: usize,
        by_id: &HashMap<&str, &Value>,
        live: &mut Live,
        interactive_only: bool,
        out: &mut String,
        lines: &mut usize,
    ) {
        if *lines >= 400 {
            return;
        }
        let Some(node) = by_id.get(id) else { return };
        // Ignored/container nodes get no line, but their children walk on.
        let transparent =
            node["ignored"].as_bool() == Some(true) || SKIP_ROLES.contains(&value(node, "role"));
        let role = value(node, "role");
        let name = value(node, "name");
        let val = value(node, "value");
        let interactive = INTERACTIVE_ROLES.contains(&role);
        let show = !transparent
            && (interactive || (!interactive_only && !name.is_empty()))
            && (!name.is_empty() || !val.is_empty() || interactive);
        if show {
            let mut line = format!("{:indent$}{role}", "", indent = depth * 2);
            if !name.is_empty() {
                line.push_str(&format!(" \"{name}\""));
            }
            if !val.is_empty() {
                line.push_str(&format!(" = \"{val}\""));
            }
            if let Some(backend) = interactive
                .then(|| node["backendDOMNodeId"].as_i64())
                .flatten()
            {
                live.next_ref += 1;
                live.refs.insert(live.next_ref, backend);
                line.push_str(&format!(" [e{}]", live.next_ref));
            }
            out.push_str(&line);
            out.push('\n');
            *lines += 1;
        }
        let deeper = if transparent { depth } else { depth + 1 };
        for child in node["childIds"].as_array().into_iter().flatten() {
            if let Some(cid) = child.as_str() {
                walk(cid, deeper, by_id, live, interactive_only, out, lines);
            }
        }
    }

    if let Some(root) = nodes.first()
        && let Some(id) = root["nodeId"].as_str()
    {
        walk(id, 0, &by_id, live, interactive_only, &mut out, &mut 0);
    }
    if out.is_empty() {
        "empty page".to_string()
    } else {
        out
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RefArgs {
    /// Element ref from `browser_snapshot` (the `eN` tag).
    r#ref: u32,
}

browser_tool!(
    Click,
    "browser_click",
    "Click an element by its snapshot ref (eN). The element is scrolled \
     into view and clicked with real mouse events at a slightly jittered \
     point.",
    RefArgs,
    |live, args| {
        let (x, y) = live.point_at(args.r#ref).await?;
        live.click_at(x, y).await?;
        Ok(ToolResult::text(format!("clicked e{}", args.r#ref)))
    }
);

browser_tool!(
    Hover,
    "browser_hover",
    "Move the mouse over an element by its snapshot ref (eN).",
    RefArgs,
    |live, args| {
        let (x, y) = live.point_at(args.r#ref).await?;
        let (jx, jy) = jitter();
        live.mouse("mouseMoved", x + jx, y + jy).await?;
        Ok(ToolResult::text(format!("hovered e{}", args.r#ref)))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct TypeArgs {
    /// Element ref from `browser_snapshot`.
    r#ref: u32,
    /// The text to enter.
    text: String,
    /// Clear the field first (default false).
    clear: Option<bool>,
}

browser_tool!(
    TypeText,
    "browser_type",
    "Focus an element (eN) and type text into it as real input. \
     `clear: true` empties the field first.",
    TypeArgs,
    |live, args| {
        let backend = live.backend(args.r#ref)?;
        live.call("DOM.focus", json!({"backendNodeId": backend}))
            .await?;
        if args.clear.unwrap_or(false) {
            let resolved = live
                .call("DOM.resolveNode", json!({"backendNodeId": backend}))
                .await?;
            if let Some(object) = resolved["object"]["objectId"].as_str() {
                live.call(
                    "Runtime.callFunctionOn",
                    json!({
                        "objectId": object,
                        "functionDeclaration": "function(){this.value='';this.dispatchEvent(new Event('input',{bubbles:true}))}",
                    }),
                )
                .await?;
            }
        }
        live.call("Input.insertText", json!({"text": args.text}))
            .await?;
        Ok(ToolResult::text(format!("typed into e{}", args.r#ref)))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct PressKeyArgs {
    /// Key name: Enter, Tab, Escape, Backspace, Delete, Space,
    /// ArrowLeft/Right/Up/Down, Home, End, PageUp, PageDown — or a
    /// single character.
    key: String,
    /// Held modifiers: any of "ctrl", "shift", "alt", "meta".
    modifiers: Option<Vec<String>>,
}

/// (key, code, windowsVirtualKeyCode, text) for named keys.
fn key_spec(key: &str) -> Option<(&'static str, &'static str, i64, &'static str)> {
    Some(match key {
        "Enter" => ("Enter", "Enter", 13, "\r"),
        "Tab" => ("Tab", "Tab", 9, ""),
        "Escape" | "Esc" => ("Escape", "Escape", 27, ""),
        "Backspace" => ("Backspace", "Backspace", 8, ""),
        "Delete" => ("Delete", "Delete", 46, ""),
        " " | "Space" => (" ", "Space", 32, " "),
        "ArrowLeft" => ("ArrowLeft", "ArrowLeft", 37, ""),
        "ArrowUp" => ("ArrowUp", "ArrowUp", 38, ""),
        "ArrowRight" => ("ArrowRight", "ArrowRight", 39, ""),
        "ArrowDown" => ("ArrowDown", "ArrowDown", 40, ""),
        "Home" => ("Home", "Home", 36, ""),
        "End" => ("End", "End", 35, ""),
        "PageUp" => ("PageUp", "PageUp", 33, ""),
        "PageDown" => ("PageDown", "PageDown", 34, ""),
        _ => return None,
    })
}

browser_tool!(
    PressKey,
    "browser_press_key",
    "Press a key as real keyboard events — Enter, Tab, Escape, arrows, \
     Backspace, or a single character, with optional ctrl/shift/alt/meta \
     modifiers.",
    PressKeyArgs,
    |live, args| {
        let modifiers: i64 = args
            .modifiers
            .unwrap_or_default()
            .iter()
            .map(|m| match m.as_str() {
                "alt" => 1,
                "ctrl" => 2,
                "meta" => 4,
                "shift" => 8,
                _ => 0,
            })
            .sum();
        let (key, code, vk, text): (String, String, i64, String) = match key_spec(&args.key) {
            Some((k, c, v, t)) => (k.to_string(), c.to_string(), v, t.to_string()),
            None if args.key.chars().count() == 1 => {
                let ch = args.key.clone();
                let c = ch.chars().next().unwrap_or(' ');
                let upper = c.to_ascii_uppercase();
                let code = if c.is_ascii_alphabetic() {
                    format!("Key{upper}")
                } else if c.is_ascii_digit() {
                    format!("Digit{c}")
                } else {
                    String::new()
                };
                (ch.clone(), code, upper as i64, ch)
            }
            None => {
                return Err(BrowserError::Protocol(format!(
                    "unknown key {:?} — use a named key or single character",
                    args.key
                )));
            }
        };
        for kind in ["rawKeyDown", "keyUp"] {
            let mut params = json!({
                "type": kind,
                "key": key,
                "code": code,
                "windowsVirtualKeyCode": vk,
                "nativeVirtualKeyCode": vk,
                "modifiers": modifiers,
            });
            if !text.is_empty() && kind == "rawKeyDown" {
                params["text"] = json!(text);
            }
            live.call("Input.dispatchKeyEvent", params).await?;
        }
        Ok(ToolResult::text(format!("pressed {}", args.key)))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct ScrollArgs {
    /// "up" or "down" (default "down").
    direction: Option<String>,
    /// Pixels to scroll (default 600).
    amount: Option<i64>,
    /// Or an element ref to scroll into view instead.
    r#ref: Option<u32>,
}

browser_tool!(
    Scroll,
    "browser_scroll",
    "Scroll the page (mouse wheel) or scroll an element into view by ref.",
    ScrollArgs,
    |live, args| {
        if let Some(reference) = args.r#ref {
            live.point_at(reference).await?;
            return Ok(ToolResult::text(format!("scrolled e{reference} into view")));
        }
        let delta = args.amount.unwrap_or(600)
            * if args.direction.as_deref() == Some("up") {
                -1
            } else {
                1
            };
        live.call(
            "Input.dispatchMouseEvent",
            json!({"type": "mouseWheel", "x": 400.0, "y": 300.0, "deltaX": 0, "deltaY": delta}),
        )
        .await?;
        Ok(ToolResult::text(format!("scrolled {delta}px")))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct EvaluateArgs {
    /// JavaScript to run — an expression or `(() => {...})()`. Runs in
    /// an isolated world: the page can't see your globals. The result
    /// is returned by value (awaited if a Promise).
    expression: String,
}

browser_tool!(
    Evaluate,
    "browser_evaluate",
    "Run JavaScript in the page (isolated world — your globals never \
     touch the page's). Returns the result by value; Promises are \
     awaited. For reading page structure prefer browser_snapshot.",
    EvaluateArgs,
    |live, args| {
        let value = live.eval_isolated(&args.expression).await?;
        Ok(ToolResult::text(match value {
            Value::String(s) => s,
            other => serde_json::to_string_pretty(&other).unwrap_or_default(),
        }))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct ScreenshotArgs {
    /// Capture the full scrollable page, not just the viewport.
    full_page: Option<bool>,
}

browser_tool!(
    Screenshot,
    "browser_screenshot",
    "Take a PNG screenshot of the current page (viewport, or the whole \
     scrollable page with full_page). Saved under the daemon's \
     browser/ dir and returned inline.",
    ScreenshotArgs,
    |live, args| {
        let shot = live
            .call(
                "Page.captureScreenshot",
                json!({
                    "format": "png",
                    "captureBeyondViewport": args.full_page.unwrap_or(false),
                }),
            )
            .await?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(shot["data"].as_str().unwrap_or_default())
            .map_err(|e| BrowserError::Protocol(format!("bad screenshot: {e}")))?;
        let path = live.out_dir.join(format!(
            "shot-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        std::fs::write(&path, &bytes).map_err(BrowserError::Io)?;
        Ok(ToolResult::parts(vec![
            ToolResultPart::text(format!("saved {}", path.display())),
            ToolResultPart::image(bytes, "image/png"),
        ]))
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct WaitForArgs {
    /// Wait until this text appears in the page body.
    text: Option<String>,
    /// Wait until this CSS selector matches.
    selector: Option<String>,
    /// Or just sleep this many milliseconds.
    time_ms: Option<u64>,
    /// Give up after this long (default 10s, cap 60s).
    timeout_ms: Option<u64>,
}

browser_tool!(
    WaitFor,
    "browser_wait_for",
    "Wait until text appears in the page, a CSS selector matches, or a \
     fixed time passes. Errors on timeout.",
    WaitForArgs,
    |live, args| {
        let timeout =
            std::time::Duration::from_millis(args.timeout_ms.unwrap_or(10_000).min(60_000));
        if let Some(ms) = args.time_ms {
            async_io::Timer::after(std::time::Duration::from_millis(ms.min(60_000))).await;
            return Ok(ToolResult::text(format!("waited {ms}ms")));
        }
        let probe = match (&args.text, &args.selector) {
            (Some(text), _) => format!(
                "document.body && document.body.innerText.includes({})",
                serde_json::to_string(text).unwrap_or_default()
            ),
            (None, Some(sel)) => format!(
                "!!document.querySelector({})",
                serde_json::to_string(sel).unwrap_or_default()
            ),
            (None, None) => {
                return Err(BrowserError::Protocol(
                    "pass `text`, `selector`, or `time_ms`".to_string(),
                ));
            }
        };
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if live.eval(&probe).await.ok().and_then(|v| v.as_bool()) == Some(true) {
                return Ok(ToolResult::text("condition met".to_string()));
            }
            if std::time::Instant::now() > deadline {
                return Err(BrowserError::Timeout("browser_wait_for".to_string()));
            }
            async_io::Timer::after(std::time::Duration::from_millis(150)).await;
        }
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct TabsArgs {
    /// `list` (default), `new`, `select`, or `close`.
    action: Option<String>,
    /// Tab index for `select`/`close` (from `list`); default = current.
    index: Option<usize>,
    /// URL for `new` (default about:blank).
    url: Option<String>,
}

browser_tool!(
    Tabs,
    "browser_tabs",
    "Manage browser tabs: `list`, `new` (optional url), `select` \
     (index), `close` (index, default current).",
    TabsArgs,
    |live, args| {
        match args.action.as_deref().unwrap_or("list") {
            "list" => {
                let targets = live.targets().await?;
                let mut out = String::new();
                for (i, (id, title, url)) in targets.iter().enumerate() {
                    let mark = if id == &live.target {
                        " ← current"
                    } else {
                        ""
                    };
                    out.push_str(&format!("[{i}] {title} — {url}{mark}\n"));
                }
                Ok(ToolResult::text(if out.is_empty() {
                    "no tabs".to_string()
                } else {
                    out
                }))
            }
            "new" => {
                let url = args.url.as_deref().unwrap_or("about:blank");
                let (id, ..) = live.new_target(url).await?;
                live.attach(&id).await?;
                Ok(ToolResult::text(format!("opened tab {url}")))
            }
            "select" => {
                let targets = live.targets().await?;
                let index = args.index.unwrap_or(0);
                let Some((id, ..)) = targets.get(index) else {
                    return Err(BrowserError::Protocol(format!("no tab {index}")));
                };
                live.attach(id).await?;
                live.cdp
                    .call(None, "Target.activateTarget", json!({"targetId": id}))
                    .await?;
                Ok(ToolResult::text(format!("selected tab {index}")))
            }
            "close" => {
                let targets = live.targets().await?;
                let index = args.index.unwrap_or_else(|| {
                    targets
                        .iter()
                        .position(|(id, ..)| id == &live.target)
                        .unwrap_or(0)
                });
                let Some((id, ..)) = targets.get(index) else {
                    return Err(BrowserError::Protocol(format!("no tab {index}")));
                };
                let closing_current = id == &live.target;
                live.cdp
                    .call(None, "Target.closeTarget", json!({"targetId": id}))
                    .await?;
                if closing_current {
                    live.ensure_target().await?;
                }
                Ok(ToolResult::text(format!("closed tab {index}")))
            }
            other => Err(BrowserError::Protocol(format!(
                "unknown tabs action {other:?}"
            ))),
        }
    }
);

browser_tool!(
    NavigateBack,
    "browser_navigate_back",
    "Go back one step in the current tab's history.",
    EmptyArgs,
    |live, _args| {
        let history = live.call("Page.getNavigationHistory", json!({})).await?;
        let index = history["currentIndex"].as_i64().unwrap_or(0);
        let entry = history["entries"]
            .as_array()
            .and_then(|e| e.get(usize::try_from(index - 1).unwrap_or(0)));
        match entry.and_then(|e| e["id"].as_i64()) {
            Some(id) if index > 0 => {
                live.call("Page.navigateToHistoryEntry", json!({"entryId": id}))
                    .await?;
                live.refs.clear();
                live.wait_loaded(std::time::Duration::from_secs(30)).await?;
                let (title, url) = live.location().await;
                Ok(ToolResult::text(format!("{title}\n{url}")))
            }
            _ => Ok(ToolResult::text("no history to go back to")),
        }
    }
);

#[derive(Debug, Deserialize, JsonSchema)]
struct LimitArgs {
    /// Max entries to return (default 50).
    limit: Option<usize>,
}

browser_tool!(
    ConsoleLog,
    "browser_console",
    "Console messages logged by the page since the last call (errors, \
     warnings, logs — drained, capped at 200).",
    LimitArgs,
    |live, args| {
        let limit = args.limit.unwrap_or(50);
        let mut buffers = live.cdp.buffers.lock().await;
        let mut out = String::new();
        while let Some((level, text, url)) = buffers.console.pop_front() {
            out.push_str(&format!("[{level}] {text} ({url})\n"));
            if out.lines().count() >= limit {
                break;
            }
        }
        Ok(ToolResult::text(if out.is_empty() {
            "no console messages".to_string()
        } else {
            out
        }))
    }
);

browser_tool!(
    NetworkLog,
    "browser_network",
    "Network requests completed since the last call: `status METHOD url` \
     lines (drained, capped at 200).",
    LimitArgs,
    |live, args| {
        let limit = args.limit.unwrap_or(50);
        let mut buffers = live.cdp.buffers.lock().await;
        let mut out = String::new();
        while let Some(line) = buffers.network.pop_front() {
            out.push_str(&line);
            out.push('\n');
            if out.lines().count() >= limit {
                break;
            }
        }
        Ok(ToolResult::text(if out.is_empty() {
            "no network requests".to_string()
        } else {
            out
        }))
    }
);

/// `browser_close` doesn't go through the macro — closing must not
/// launch a browser that isn't running.
struct CloseBrowser {
    browser: Browser,
}

impl CloseBrowser {
    fn new(browser: Browser) -> Self {
        Self { browser }
    }
}

impl Tool for CloseBrowser {
    type Arguments = EmptyArgs;
    type Res = ToolResult;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "browser_close".into()
    }
    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Kill the browser entirely (cookies and logins stay on disk — the \
         profile persists). The next browser_* call relaunches fresh."
            .into()
    }
    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<Self::Res> {
        let mut state = self.browser.inner.lock().await;
        *state = State::Down; // dropping Live kills the child
        Ok(ToolResult::text(
            "browser closed — profile kept; next call relaunches",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_all_browser_tools() {
        let browser = Browser::new(BrowserConfig::default(), PathBuf::from("/tmp/x"));
        let mut tools = Tools::new();
        browser.register_into(&mut tools);
        let names: Vec<String> = tools
            .definitions()
            .iter()
            .map(|d| d.name().to_string())
            .collect();
        for expected in [
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_press_key",
            "browser_hover",
            "browser_scroll",
            "browser_evaluate",
            "browser_screenshot",
            "browser_wait_for",
            "browser_tabs",
            "browser_navigate_back",
            "browser_console",
            "browser_network",
            "browser_close",
        ] {
            assert!(names.iter().any(|n| n == expected), "missing {expected}");
        }
    }

    /// `executor_core::spawn` (the CDP reader task) needs a global
    /// executor — leak one, unless a sibling test already did.
    fn ensure_executor() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let executor: &'static async_executor::Executor<'static> =
                Box::leak(Box::new(async_executor::Executor::new()));
            if executor_core::try_init_global_executor(executor).is_err() {
                return;
            }
            for _ in 0..2 {
                std::thread::spawn(move || {
                    futures_lite::future::block_on(executor.run(std::future::pending::<()>()));
                });
            }
        });
    }

    /// End-to-end against a real Chrome: launch, navigate, snapshot,
    /// evaluate, screenshot. `cargo test -- --ignored real_chrome`.
    #[test]
    #[ignore = "needs a Chrome-family browser installed"]
    fn real_chrome_end_to_end() {
        ensure_executor();
        let dir = std::env::temp_dir().join("acpbot-browser-test");
        let _ = std::fs::remove_dir_all(&dir);
        futures_lite::future::block_on(async {
            let browser = Browser::new(BrowserConfig::default(), dir.clone());
            let mut tools = Tools::new();
            browser.register_into(&mut tools);

            let result = tools
                .call("browser_navigate", r#"{"url":"https://example.com"}"#)
                .await
                .expect("navigate call");
            let text = result.as_text().unwrap_or_default().to_string();
            assert!(text.contains("Example Domain"), "got: {text}");

            let result = tools
                .call("browser_snapshot", r#"{}"#)
                .await
                .expect("snapshot call");
            let text = result.as_text().unwrap_or_default().to_string();
            assert!(text.contains("link"), "snapshot: {text}");

            let result = tools
                .call(
                    "browser_evaluate",
                    r#"{"expression":"navigator.webdriver"}"#,
                )
                .await
                .expect("evaluate call");
            assert_eq!(result.as_text().unwrap_or_default(), "false");

            let result = tools
                .call("browser_screenshot", r#"{}"#)
                .await
                .expect("screenshot call");
            let _ = result;

            let _ = std::fs::remove_dir_all(&dir);
        });
    }
}
