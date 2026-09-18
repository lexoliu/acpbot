//! Link previews for inbound message text.
//!
//! When an event's `text` carries links the daemon fetches each one and
//! attaches an [`EventLinkPreview`] to the event before it is prompted —
//! the agent reads `link_previews` instead of spending turns on its own
//! fetches. The fetch is prompt-assembly decoration, like `media.file`:
//! it runs inside `turn`, never reaches the in-flight journal or IM
//! history, and replays re-fetch (or hit the short-lived cache).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_lite::StreamExt;
use tracing::warn;
use zenwave::Client;

use crate::chat::{ChatEvent, EventLinkPreview};
use crate::config::PreviewConfig;

/// Bodies are read only far enough to cover the head — meta tags never
/// live past it, and a cap keeps a hostile page from making the daemon
/// buffer unbounded HTML.
const MAX_BODY: usize = 512 * 1024;
/// Title/description lengths a chat message can usefully carry.
const TITLE_MAX: usize = 300;
const TEXT_MAX: usize = 1200;
/// Identical links re-posted inside this window reuse the first fetch.
const CACHE_TTL: Duration = Duration::from_secs(600);
/// Hard bound on cache growth; a full cache is cleared, not evicted —
/// link volume is low and churn is cheap.
const CACHE_MAX: usize = 256;

/// A desktop-Chrome UA: pages that fingerprint clients answer this one
/// with the real markup — including `og:` tags some sites only serve to
/// plausible browsers.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

/// The link-preview fetcher, shared across events: per-event URL
/// concurrency, a per-link timeout, and a small TTL cache.
pub struct Previewer {
    timeout: Duration,
    max_links: usize,
    cache: Arc<Mutex<HashMap<String, (Instant, EventLinkPreview)>>>,
}

impl Previewer {
    pub fn new(config: &PreviewConfig) -> Self {
        Self {
            timeout: Duration::from_secs(config.timeout_secs),
            max_links: config.max_links,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Attach `link_previews` for every link in `event.text` (at most
    /// `max_links`, fetched concurrently). No-op when the text carries
    /// no fetchable URLs.
    pub async fn enrich(&self, event: &mut ChatEvent) {
        let Some(text) = &event.text else { return };
        let links = extract_links(text, self.max_links);
        if links.is_empty() {
            return;
        }
        let mut previews: Vec<Option<EventLinkPreview>> = vec![None; links.len()];
        let mut tasks = Vec::new();
        for (index, url) in links.iter().enumerate() {
            if let Some(preview) = self.cached(url).or_else(|| short_circuit(url)) {
                previews[index] = Some(preview);
                continue;
            }
            let url = url.clone();
            let cache = Arc::clone(&self.cache);
            let timeout = self.timeout;
            tasks.push((
                index,
                executor_core::spawn(async move {
                    let preview = fetch(&url, timeout).await;
                    let mut cache = cache.lock().expect("preview cache poisoned");
                    if cache.len() >= CACHE_MAX {
                        cache.clear();
                    }
                    cache.insert(url, (Instant::now(), preview.clone()));
                    preview
                }),
            ));
        }
        for (index, task) in tasks {
            previews[index] = Some(task.await);
        }
        event.link_previews = previews.into_iter().flatten().collect();
    }

    fn cached(&self, url: &str) -> Option<EventLinkPreview> {
        self.cache
            .lock()
            .expect("preview cache poisoned")
            .get(url)
            .filter(|(fetched, _)| fetched.elapsed() < CACHE_TTL)
            .map(|(_, preview)| preview.clone())
    }
}

/// The `http(s)` links in a message — scheme-qualified only (a bare
/// `t.me/x` without a scheme isn't fetchable as posted), deduplicated,
/// capped at `max`.
fn extract_links(text: &str, max: usize) -> Vec<String> {
    let mut finder = linkify::LinkFinder::new();
    finder.kinds(&[linkify::LinkKind::Url]);
    let mut seen: Vec<String> = Vec::new();
    for link in finder.links(text) {
        let url = link.as_str();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            continue;
        }
        if !seen.iter().any(|u| u == url) {
            seen.push(url.to_string());
        }
        if seen.len() >= max {
            break;
        }
    }
    seen
}

/// Links answered without a fetch — private Telegram URLs, and
/// loopback/private hosts the daemon must never touch: messages come
/// from chat users while fetches run on the host network (SSRF).
fn short_circuit(url: &str) -> Option<EventLinkPreview> {
    let uri: zenwave::Uri = url.try_into().ok()?;
    let host = uri
        .host()?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    let path = uri.path();
    let error = if telegram_host(&host)
        && (path.starts_with("/c/") || path.starts_with("/+") || path.starts_with("/joinchat"))
    {
        "private Telegram link — only resolvable inside Telegram".to_string()
    } else if host == "localhost"
        || host.strip_suffix(".local").is_some()
        || host.parse::<std::net::IpAddr>().is_ok_and(is_private)
    {
        format!("{host} is a private host — not fetched")
    } else {
        return None;
    };
    Some(EventLinkPreview {
        url: url.to_string(),
        site: None,
        title: None,
        text: None,
        error: Some(error),
    })
}

fn is_private(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
        }
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
    }
}

fn telegram_host(host: &str) -> bool {
    matches!(
        host,
        "t.me" | "telegram.me" | "telegram.dog" | "www.t.me" | "www.telegram.me"
    )
}

/// For a `t.me/<name>/<post>` link: the `/s/` URL to fetch (the plain
/// post page needs JS; `/s/` is server-rendered) plus the `data-post`
/// id (`name/post`) marking the widget to read. `t.me/s/...` links keep
/// their form. Anything else returns `None` and fetches as posted.
fn telegram_target(url: &str) -> Option<(String, String)> {
    let uri: zenwave::Uri = url.try_into().ok()?;
    let host = uri.host()?.to_ascii_lowercase();
    if !telegram_host(&host) {
        return None;
    }
    let path = uri.path().trim_matches('/');
    let path = path.strip_prefix("s/").unwrap_or(path);
    let mut parts = path.split('/');
    let (Some(name), Some(id), None) = (parts.next(), parts.next(), parts.next()) else {
        return None;
    };
    if name.starts_with('+') || !id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let name = name.to_lowercase();
    Some((
        format!("https://t.me/s/{name}/{id}"),
        format!("{name}/{id}"),
    ))
}

async fn fetch(url: &str, timeout: Duration) -> EventLinkPreview {
    match try_fetch(url, timeout).await {
        Ok(preview) => preview,
        Err(error) => {
            warn!(%url, %error, "link preview fetch failed");
            EventLinkPreview {
                url: url.to_string(),
                site: None,
                title: None,
                text: None,
                error: Some(error.to_string()),
            }
        }
    }
}

/// Redirect hops to follow by hand. `raw_client` is used instead of the
/// default redirect middleware so every hop's `Location` target goes
/// through [`short_circuit`] — otherwise a posted URL could redirect to
/// a private host the initial check never saw.
const MAX_REDIRECTS: usize = 5;

async fn try_fetch(url: &str, timeout: Duration) -> Result<EventLinkPreview, String> {
    let (mut target, post) = telegram_target(url)
        .map(|(u, p)| (u, Some(p)))
        .unwrap_or_else(|| (url.to_string(), None));
    let mut client = zenwave::raw_client().timeout(timeout);
    for _ in 0..MAX_REDIRECTS {
        let response = client
            .get(target.as_str())
            .map_err(|e| e.to_string())?
            .header("User-Agent", USER_AGENT)
            .map_err(|e| e.to_string())?
            .header("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
            .map_err(|e| e.to_string())?
            .header("Accept-Language", "en-US,en;q=0.9")
            .map_err(|e| e.to_string())?
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        if status.is_redirection()
            && let Some(location) = response
                .headers()
                .get(zenwave::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        {
            target = url::Url::parse(&target)
                .and_then(|base| base.join(&location))
                .map_err(|e| e.to_string())?
                .to_string();
            if let Some(preview) = short_circuit(&target) {
                return Ok(preview);
            }
            continue;
        }
        if !status.is_success() {
            return Err(format!("HTTP {status}"));
        }
        let content_type = response
            .headers()
            .get(zenwave::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let mut body = response.into_body();
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            bytes.extend_from_slice(&chunk);
            if bytes.len() > MAX_BODY {
                break;
            }
        }
        let html = String::from_utf8_lossy(&bytes);
        return Ok(parse(&html, url, &content_type, post.as_deref()));
    }
    Err("redirect loop".to_string())
}

/// Extract the preview from fetched markup. Non-HTML bodies surface as
/// an `error` naming the media type — "this link is a 4 MB image/png"
/// is itself a useful preview.
fn parse(html: &str, url: &str, content_type: &str, post: Option<&str>) -> EventLinkPreview {
    let mut preview = EventLinkPreview {
        url: url.to_string(),
        site: None,
        title: None,
        text: None,
        error: None,
    };
    if !content_type.contains("html") {
        preview.error = Some(format!("not a web page ({content_type})"));
        return preview;
    }
    let Ok(dom) = tl::parse(html, tl::ParserOptions::default()) else {
        preview.error = Some("unparsable HTML".to_string());
        return preview;
    };
    let parser = dom.parser();
    let meta = |selector: &str| {
        dom.query_selector(selector)
            .and_then(|mut nodes| nodes.next())
            .and_then(|handle| handle.get(parser))
            .and_then(|node| {
                node.as_tag()
                    .and_then(|tag| tag.attributes().get("content").flatten())
                    .map(|raw| clean(raw.as_utf8_str().as_ref(), TEXT_MAX))
            })
            .filter(|text| !text.is_empty())
    };
    preview.site = meta("meta[property='og:site_name']");
    preview.title = meta("meta[property='og:title']").or_else(|| {
        dom.query_selector("title")
            .and_then(|mut nodes| nodes.next())
            .and_then(|handle| handle.get(parser))
            .map(|node| clean(node.inner_text(parser).as_ref(), TITLE_MAX))
            .filter(|text| !text.is_empty())
    });
    // A Telegram post's own body beats the channel bio that
    // `og:description` carries on `/s/` pages.
    preview.text = post
        .and_then(|post| telegram_post_text(&dom, post))
        .or_else(|| meta("meta[property='og:description']"))
        .or_else(|| meta("meta[name='description']"));
    if preview.title.is_none() && preview.text.is_none() {
        preview.error = Some("page exposes no title or description".to_string());
    }
    preview
}

/// The post body on a `t.me/s/…` page: the `tgme_widget_message_text`
/// div inside the widget whose `data-post` is `<name>/<id>` — the page
/// renders a window of posts, so the attribute match picks the one the
/// link pointed at.
fn telegram_post_text(dom: &tl::VDom<'_>, post: &str) -> Option<String> {
    let widget = dom
        .nodes()
        .iter()
        .filter_map(tl::Node::as_tag)
        .find(|tag| {
            tag.attributes()
                .get("data-post")
                .flatten()
                .is_some_and(|v| v.as_utf8_str().eq_ignore_ascii_case(post))
        })?;
    widget
        .query_selector(dom.parser(), "div.tgme_widget_message_text")
        .and_then(|mut nodes| nodes.next())
        .and_then(|handle| handle.get(dom.parser()))
        .map(|node| clean(node.inner_text(dom.parser()).as_ref(), TEXT_MAX))
        .filter(|text| !text.is_empty())
}

/// Collapse whitespace and decode entities — preview strings go into a
/// JSON chat message, not into markup.
fn clean(raw: &str, max: usize) -> String {
    let decoded = html_escape::decode_html_entities(raw);
    decoded
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_links_finds_scheme_urls_deduped() {
        let links = extract_links(
            "look https://t.me/durov/236 and http://a.example.org/c?x=1 again https://t.me/durov/236 \
             and bare example.org plus mailto:a@b.c",
            10,
        );
        assert_eq!(
            links,
            vec![
                "https://t.me/durov/236".to_string(),
                "http://a.example.org/c?x=1".to_string()
            ]
        );
        assert!(extract_links("no links here", 3).is_empty());
        // The cap binds.
        assert_eq!(
            extract_links("https://a.example.org https://c.example.org", 1).len(),
            1
        );
    }

    #[test]
    fn short_circuit_covers_private_and_local() {
        assert_eq!(
            short_circuit("https://t.me/c/1234/56")
                .unwrap()
                .error
                .as_deref(),
            Some("private Telegram link — only resolvable inside Telegram")
        );
        assert!(short_circuit("https://t.me/+abcdef").is_some());
        assert!(short_circuit("http://localhost:8080/x").is_some());
        assert!(short_circuit("http://192.168.1.1/").is_some());
        assert!(short_circuit("http://[::1]/").is_some());
        assert!(short_circuit("https://t.me/durov/236").is_none());
        assert!(short_circuit("https://example.org/page").is_none());
    }

    #[test]
    fn telegram_target_rewrites_post_links() {
        assert_eq!(
            telegram_target("https://t.me/durov/236"),
            Some((
                "https://t.me/s/durov/236".to_string(),
                "durov/236".to_string()
            ))
        );
        assert_eq!(
            telegram_target("https://t.me/s/Durov/236"),
            Some((
                "https://t.me/s/durov/236".to_string(),
                "durov/236".to_string()
            ))
        );
        // A bare channel link fetches as posted (bio in og:description).
        assert_eq!(telegram_target("https://t.me/durov"), None);
        assert_eq!(telegram_target("https://t.me/durov/236/comments"), None);
        assert_eq!(telegram_target("https://example.org/a/1"), None);
    }

    #[test]
    fn parse_reads_og_meta_and_title() {
        let html = r#"<!DOCTYPE html><html><head>
            <title>Fallback Title</title>
            <meta property="og:title" content="OG &amp; Title">
            <meta property="og:site_name" content="Site">
            <meta property="og:description" content="A   description
                with newlines">
            </head><body></body></html>"#;
        let p = parse(html, "https://x.y/", "text/html; charset=utf-8", None);
        assert_eq!(p.site.as_deref(), Some("Site"));
        assert_eq!(p.title.as_deref(), Some("OG & Title"));
        assert_eq!(p.text.as_deref(), Some("A description with newlines"));
        assert!(p.error.is_none());

        let bare = parse(
            "<title>Only Title</title>",
            "https://x.y/",
            "text/html",
            None,
        );
        assert_eq!(bare.title.as_deref(), Some("Only Title"));
        assert!(bare.text.is_none());
        assert!(bare.error.is_none());

        let img = parse("", "https://x.y/i.png", "image/png", None);
        assert_eq!(img.error.as_deref(), Some("not a web page (image/png)"));

        let empty = parse("<html></html>", "https://x.y/", "text/html", None);
        assert_eq!(
            empty.error.as_deref(),
            Some("page exposes no title or description")
        );
    }

    #[test]
    fn telegram_post_text_picks_the_linked_post() {
        let html = r#"<div class="tgme_widget_message" data-post="chan/1">
                <div class="tgme_widget_message_text">wrong post</div></div>
            <div class="tgme_widget_message" data-post="chan/2">
                <div class="tgme_widget_message_text js-message_text">the <b>linked</b> post &quot;yes&quot;</div></div>
            <div class="tgme_widget_message" data-post="chan/3">
                <div class="tgme_widget_message_text">later post</div></div>"#;
        let dom = tl::parse(html, tl::ParserOptions::default()).unwrap();
        assert_eq!(
            telegram_post_text(&dom, "chan/2").as_deref(),
            Some("the linked post \"yes\"")
        );
        assert!(telegram_post_text(&dom, "chan/9").is_none());
    }

    /// `enrich` populates `link_previews` on the event — here entirely
    /// via short-circuits, so no executor or network is needed.
    #[test]
    fn enrich_attaches_previews_for_links_only() {
        let mut event = ChatEvent {
            kind: "message".into(),
            platform: "telegram".into(),
            chat: "1".to_string(),
            ts: 0,
            attention: "direct".into(),
            chat_type: None,
            chat_title: None,
            message_id: None,
            from: crate::chat::EventSender {
                id: "7".to_string(),
                name: "Ada".to_string(),
            },
            text: Some(
                "see https://t.me/c/1/2 and http://127.0.0.1:9/x — no fetch happens".to_string(),
            ),
            command: None,
            button: None,
            reply_to: None,
            sticker: None,
            media: None,
            reaction: None,
            watch: None,
            link_previews: Vec::new(),
            thread_id: None,
        };
        let previewer = Previewer::new(&PreviewConfig::default());
        futures_lite::future::block_on(previewer.enrich(&mut event));
        assert_eq!(event.link_previews.len(), 2);
        assert!(
            event.link_previews.iter().all(|p| p.error.is_some()),
            "{:?}",
            event.link_previews
        );

        event.text = Some("no links".to_string());
        event.link_previews.clear();
        futures_lite::future::block_on(previewer.enrich(&mut event));
        assert!(event.link_previews.is_empty());
        event.text = None;
        futures_lite::future::block_on(previewer.enrich(&mut event));
        assert!(event.link_previews.is_empty());
    }

    /// The real path end to end: a public Telegram post and a generic
    /// page. Needs network; run with --ignored.
    #[test]
    #[ignore = "needs network access"]
    fn live_fetches_telegram_post_and_og_page() {
        let rt = futures_lite::future::block_on;
        let post = rt(fetch("https://t.me/durov/236", Duration::from_secs(15)));
        assert!(post.error.is_none(), "{:?}", post.error);
        assert!(post.title.is_some(), "{post:?}");
        assert!(post.text.is_some(), "{post:?}");
        let page = rt(fetch(
            "https://github.com/lexoliu/acpbot",
            Duration::from_secs(15),
        ));
        assert!(page.title.is_some(), "{page:?}");
    }
}
