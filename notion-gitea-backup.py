#!/usr/bin/env python3
"""
Notion → Gitea Backup
=====================
Backs up Notion databases to self-hosted Gitea repositories.
Incremental sync, AI-powered image naming, Discord notifications.

Zero external dependencies — uses only Python 3 standard library.

Usage:
    python3 notion-gitea-sync.py                    # Normal run
    python3 notion-gitea-sync.py --full-sync        # Force full re-sync
    python3 notion-gitea-sync.py --dry-run          # Preview without committing
    python3 notion-gitea-sync.py --env /path/to/.env

Configuration:
    Copy .env.example to .env and fill in your values.
    See README.md for full setup instructions.
"""

import json
import base64
import ssl
import re
import os
import sys
import time
import logging
import signal
import socket
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("notion-gitea-sync")

# Ensure Ctrl+C works even during blocked I/O
def _handle_sigint(*_):
    print("\nInterrupted by user (Ctrl+C). Exiting.", file=sys.stderr)
    os._exit(130)

signal.signal(signal.SIGINT, _handle_sigint)

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
STATE_FILE = SCRIPT_DIR / "sync-state.json"
NOTION_VERSION = "2022-06-28"

# SSL context that accepts self-signed certs (for internal Gitea)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def load_env(env_path: str = None) -> dict:
    """Parse a .env file into a dict. Skips comments and blank lines."""
    path = Path(env_path) if env_path else SCRIPT_DIR / ".env"

    if not path.exists():
        log.error(f".env file not found: {path}")
        log.error("Copy .env.example to .env and fill in your values:")
        log.error(f"  cp {SCRIPT_DIR / '.env.example'} {path}")
        sys.exit(1)

    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            env[key] = value

    return env


def build_config(env: dict) -> dict:
    """Build structured config from flat .env key-value pairs."""
    required = ["NOTION_TOKEN", "GIT_BASE_URL", "GIT_TOKEN", "GIT_OWNER"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        log.error(f"Missing required .env values: {', '.join(missing)}")
        sys.exit(1)

    # Discover databases from DB1_, DB2_, ... prefixes
    databases = []
    for i in range(1, 100):
        prefix = f"DB{i}_"
        notion_id = env.get(f"{prefix}NOTION_ID", "").strip()
        repo = env.get(f"{prefix}GIT_REPO", "").strip()
        label = env.get(f"{prefix}LABEL", "").strip()

        if not notion_id and not repo:
            if i > 1:
                break
            continue

        if not all([notion_id, repo, label]):
            log.warning(f"Incomplete database config for DB{i} — skipping. "
                        f"Need NOTION_ID, GIT_REPO, and LABEL.")
            continue

        databases.append({
            "notion_database_id": notion_id,
            "git_repo": repo,
            "label": label,
        })

    if not databases:
        log.error("No databases configured. Add DB1_NOTION_ID, DB1_GIT_REPO, "
                  "DB1_LABEL to your .env file.")
        sys.exit(1)

    # AI configuration
    ai_enabled = env.get("AI_NAMING_ENABLED", "0").strip() == "1"

    # Discord configuration: 0=never, 1=errors only, 2=always
    try:
        discord_level = int(env.get("DISCORD_LEVEL", "0").strip())
    except ValueError:
        discord_level = 0

    return {
        "notion_token": env["NOTION_TOKEN"],
        "git_base_url": env["GIT_BASE_URL"].rstrip("/"),
        "git_token": env["GIT_TOKEN"],
        "git_owner": env["GIT_OWNER"],
        "ai_enabled": ai_enabled,
        "ai_api_url": env.get("AI_API_URL", "https://api.anthropic.com/v1/messages"),
        "ai_api_key": env.get("AI_API_KEY", ""),
        "ai_model": env.get("AI_MODEL", "claude-haiku-4-5-20251001"),
        "discord_level": discord_level,
        "discord_webhook_url": env.get("DISCORD_WEBHOOK_URL", ""),
        "timezone": env.get("TIMEZONE", "UTC").strip(),
        "databases": databases,
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Migrate old format (per-database) to new format (per-page)
        if "last_sync" in data and "synced_pages" not in data:
            data = {"synced_pages": {}}
            save_state(data)
        return data
    return {"synced_pages": {}}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================
# HTTP HELPERS
# ============================================================
def http_request(url: str, method: str = "GET", headers: dict = None,
                 body: dict = None, raw: bool = False,
                 allow_404: bool = False) -> any:
    """Generic HTTP request using only standard library."""
    headers = headers or {}
    data = None

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
            content = resp.read()
            if raw:
                return content
            try:
                return json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return content.decode("utf-8", errors="replace")

    except urllib.error.HTTPError as e:
        if e.code == 404 and allow_404:
            return None
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {method} {url}: {error_body}") from e

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error {method} {url}: {e.reason}") from e


def http_download(url: str) -> bytes:
    """Download binary content, following redirects."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "NotionGitBackup/1.0",
    })
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
        return resp.read()


# ============================================================
# NOTION API
# ============================================================
class NotionClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
        }

    def _get(self, url: str) -> dict:
        time.sleep(0.34)  # Rate limit: ~3 req/sec
        return http_request(url, headers=self.headers)

    def _post(self, url: str, body: dict) -> dict:
        time.sleep(0.34)
        return http_request(url, method="POST", headers=self.headers, body=body)

    def query_database(self, db_id: str) -> list:
        """Query all pages in a database (handles pagination)."""
        pages = []
        body = {
            "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
            "page_size": 100
        }
        url = f"https://api.notion.com/v1/databases/{db_id}/query"

        while True:
            data = self._post(url, body)
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            body["start_cursor"] = data["next_cursor"]

        return pages

    def get_block_children(self, block_id: str) -> list:
        """Fetch all child blocks, including nested children recursively."""
        all_blocks = []
        cursor = None

        while True:
            url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100"
            if cursor:
                url += f"&start_cursor={cursor}"
            data = self._get(url)
            all_blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data["next_cursor"]

        for block in all_blocks:
            if block.get("has_children"):
                block["children"] = self.get_block_children(block["id"])

        return all_blocks


# ============================================================
# TEXT HELPERS
# ============================================================
def slugify(text: str, max_len: int = 40) -> str:
    import unicodedata
    # Decompose characters (e.g. ě → e + combining caron), then strip combining marks
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if len(s) <= max_len:
        return s
    # Cut at last word boundary before max_len
    truncated = s[:max_len]
    last_dash = truncated.rfind("-")
    if last_dash > max_len // 3:
        return truncated[:last_dash]
    return truncated


def rich_text_to_md(rt_arr: list) -> str:
    if not rt_arr:
        return ""
    parts = []
    for rt in rt_arr:
        text = rt.get("plain_text", "")
        if not text:
            continue
        ann = rt.get("annotations", {})
        if ann.get("code"):
            text = f"`{text}`"
        if ann.get("bold"):
            text = f"**{text}**"
        if ann.get("italic"):
            text = f"*{text}*"
        if ann.get("strikethrough"):
            text = f"~~{text}~~"
        if ann.get("underline"):
            text = f"<u>{text}</u>"
        if rt.get("href"):
            text = f"[{text}]({rt['href']})"
        parts.append(text)
    return "".join(parts)


def rich_text_to_plain(rt_arr: list) -> str:
    if not rt_arr:
        return ""
    return "".join(rt.get("plain_text", "") for rt in rt_arr)


def get_ext_from_url(url: str) -> str:
    try:
        path = urllib.parse.urlparse(url).path
        match = re.search(
            r"\.(png|jpg|jpeg|gif|webp|svg|pdf|json|ya?ml|txt|csv|xml"
            r"|sh|py|conf|cfg|toml|ini)$",
            path, re.IGNORECASE
        )
        return match.group(1).lower() if match else "png"
    except Exception:
        return "png"


def detect_media_type(data: bytes) -> str:
    """Detect image media type from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    return ""


def get_image_dimensions(data: bytes) -> tuple:
    """Extract width, height from image binary data. No external libs.
    Detects format from magic bytes, not file extension."""
    try:
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) > 24:
            w = int.from_bytes(data[16:20], 'big')
            h = int.from_bytes(data[20:24], 'big')
            return (w, h)

        # JPEG
        if data[:2] == b'\xff\xd8':
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC2):
                    h = int.from_bytes(data[i + 5:i + 7], 'big')
                    w = int.from_bytes(data[i + 7:i + 9], 'big')
                    return (w, h)
                seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
                i += 2 + seg_len
            return None

        # GIF
        if data[:6] in (b'GIF87a', b'GIF89a') and len(data) > 10:
            w = int.from_bytes(data[6:8], 'little')
            h = int.from_bytes(data[8:10], 'little')
            return (w, h)

        # WebP
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            if data[12:16] == b'VP8 ' and len(data) > 30:
                w = int.from_bytes(data[26:28], 'little') & 0x3FFF
                h = int.from_bytes(data[28:30], 'little') & 0x3FFF
                return (w, h)
            if data[12:17] == b'VP8L' and len(data) > 25:
                bits = int.from_bytes(data[21:25], 'little')
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
                return (w, h)

    except Exception:
        pass
    return None


def extract_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title" and prop.get("title"):
            return "".join(t.get("plain_text", "") for t in prop["title"])
    return "Untitled"


# ============================================================
# AI IMAGE NAMING
# ============================================================
class AiNamer:
    def __init__(self, enabled: bool, api_url: str, api_key: str,
                 model: str = "claude-haiku-4-5-20251001"):
        self.enabled = enabled
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.failures = 0
        self.errors = []

    def name_image(self, image_b64: str, ext: str, index: int,
                   heading: str, preceding_text: str,
                   media_type: str = "") -> str:
        fallback = f"{index}-{slugify(heading, 30)}.{ext}"

        if not self.enabled or not self.api_key:
            log.debug(f"  AI naming skipped (disabled or no API key), using: {fallback}")
            return fallback

        if self.failures >= 5:
            log.warning(f"  AI naming disabled (circuit breaker: {self.failures} failures), using: {fallback}")
            return fallback

        try:
            supported = {"image/png", "image/jpeg", "image/gif", "image/webp"}

            content = []
            if media_type in supported:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64
                    }
                })

            content.append({
                "type": "text",
                "text": (
                    f'Context: Image #{index} in a technical article under '
                    f'heading: "{heading}". Preceding text: '
                    f'"{preceding_text[:300]}".\n\n'
                    f"Generate a short descriptive filename (WITHOUT extension "
                    f"or number prefix). Lowercase kebab-case. Max 6 words. "
                    f"Examples: \"discord-bot-permissions-page\", "
                    f'"grafana-dashboard-cpu-metrics".\n\n'
                    f"Respond with ONLY the filename slug."
                )
            })

            log.info(f"  AI naming image {index} ({media_type}, heading: '{heading[:40]}')...")

            data = http_request(
                self.api_url,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                body={
                    "model": self.model,
                    "max_tokens": 60,
                    "messages": [{"role": "user", "content": content}],
                }
            )

            raw = (data.get("content", [{}])[0].get("text", "")).strip()
            log.info(f"  AI response: '{raw}'")

            if raw and 2 < len(raw) < 80:
                cleaned = slugify(raw, 50)
                if len(cleaned) > 2:
                    result = f"{index}-{slugify(heading, 30)}-{cleaned}.{ext}"
                    log.info(f"  AI filename: {result}")
                    return result

            log.warning(f"  AI response unusable ('{raw}'), using fallback: {fallback}")
            return fallback

        except Exception as e:
            self.failures += 1
            log.error(f"  AI naming failed for image {index} (failure {self.failures}/5): {e}")
            self.errors.append(f"AI naming failed image {index}: {e}")
            return fallback


# ============================================================
# BLOCK → MARKDOWN CONVERTER
# ============================================================
class MarkdownConverter:
    def __init__(self, ai_namer: AiNamer, page_title: str = ""):
        self.ai = ai_namer
        self.images = []
        self.attachments = []
        self.errors = []
        self.current_heading = page_title or "untitled"
        self.last_paragraph = ""

    def convert(self, blocks: list, indent: str = "") -> str:
        md = ""
        num_idx = 0
        prev_type = ""

        for block in blocks:
            btype = block.get("type", "")

            if btype != "numbered_list_item":
                num_idx = 0

            if "list_item" in prev_type and "list_item" not in btype:
                md += "\n"

            if btype in ("heading_1", "heading_2", "heading_3"):
                text = rich_text_to_plain(block.get(btype, {}).get("rich_text"))
                if text:
                    self.current_heading = text

            if btype == "paragraph":
                plain = rich_text_to_plain(
                    block.get("paragraph", {}).get("rich_text"))
                if plain.strip():
                    self.last_paragraph = plain

            md += self._convert_block(block, btype, indent, num_idx)

            if btype == "numbered_list_item":
                num_idx += 1

            prev_type = btype

        return md

    def _convert_block(self, block, btype, indent, num_idx):
        # Paragraph
        if btype == "paragraph":
            return f"{indent}{rich_text_to_md(block['paragraph'].get('rich_text'))}\n\n"

        # Headings (shifted down: Notion H1 → md H2)
        if btype == "heading_1":
            return f"## {rich_text_to_md(block['heading_1'].get('rich_text'))}\n\n"
        if btype == "heading_2":
            return f"### {rich_text_to_md(block['heading_2'].get('rich_text'))}\n\n"
        if btype == "heading_3":
            return f"#### {rich_text_to_md(block['heading_3'].get('rich_text'))}\n\n"

        # Bulleted list
        if btype == "bulleted_list_item":
            md = f"{indent}- {rich_text_to_md(block['bulleted_list_item'].get('rich_text'))}\n"
            if block.get("children"):
                md += self.convert(block["children"], indent + "    ")
            return md

        # Numbered list
        if btype == "numbered_list_item":
            md = f"{indent}{num_idx + 1}. {rich_text_to_md(block['numbered_list_item'].get('rich_text'))}\n"
            if block.get("children"):
                md += self.convert(block["children"], indent + "    ")
            return md

        # To-do
        if btype == "to_do":
            chk = "x" if block["to_do"].get("checked") else " "
            return f"{indent}- [{chk}] {rich_text_to_md(block['to_do'].get('rich_text'))}\n"

        # Code block
        if btype == "code":
            lang = block["code"].get("language", "")
            if lang == "plain text":
                lang = ""
            code = rich_text_to_md(block["code"].get("rich_text"))
            caption = rich_text_to_md(block["code"].get("caption"))
            md = f"```{lang}\n{code}\n```\n"
            if caption:
                md += f"*{caption}*\n"
            return md + "\n"

        # Image
        if btype == "image":
            return self._handle_image(block)

        # File / PDF attachment
        if btype in ("file", "pdf"):
            return self._handle_file(block, btype)

        # Divider
        if btype == "divider":
            return "---\n\n"

        # Quote
        if btype == "quote":
            text = rich_text_to_md(block["quote"].get("rich_text"))
            md = "\n".join(f"> {line}" for line in text.split("\n")) + "\n"
            if block.get("children"):
                child_md = self.convert(block["children"])
                md += "\n".join(f"> {line}" for line in child_md.split("\n")) + "\n"
            return md + "\n"

        # Callout
        if btype == "callout":
            icon = block["callout"].get("icon", {}).get("emoji", "\u2139\ufe0f")
            text = rich_text_to_md(block["callout"].get("rich_text"))
            md = f"> {icon} **Note**\n> \n> {text}\n"
            if block.get("children"):
                child_md = self.convert(block["children"])
                lines = [l for l in child_md.split("\n") if l.strip()]
                md += "\n".join(f"> {l}" for l in lines) + "\n"
            return md + "\n"

        # Toggle
        if btype == "toggle":
            summary = rich_text_to_md(block["toggle"].get("rich_text"))
            md = f"<details>\n<summary>{summary}</summary>\n\n"
            if block.get("children"):
                md += self.convert(block["children"])
            return md + "</details>\n\n"

        # Table
        if btype == "table":
            return self._handle_table(block)

        # Bookmark
        if btype == "bookmark":
            url = block["bookmark"].get("url", "")
            caption = rich_text_to_md(block["bookmark"].get("caption")) or url
            return f"\U0001f517 [{caption}]({url})\n\n"

        # Embed
        if btype == "embed":
            url = block["embed"].get("url", "")
            caption = rich_text_to_md(block["embed"].get("caption")) or url
            return f"\U0001f517 [{caption}]({url})\n\n"

        # Video
        if btype == "video":
            vd = block["video"]
            url = (vd.get("external", {}).get("url", "")
                   if vd.get("type") == "external"
                   else vd.get("file", {}).get("url", ""))
            return f"\U0001f3ac [Video]({url})\n\n"

        # Column list (flatten)
        if btype == "column_list":
            md = ""
            for col in block.get("children", []):
                if col.get("children"):
                    md += self.convert(col["children"], indent)
            return md

        # Synced block
        if btype == "synced_block":
            if block.get("children"):
                return self.convert(block["children"], indent)
            return ""

        # Child page / database
        if btype == "child_page":
            return f"\U0001f4c4 **Sub-page:** {block['child_page'].get('title', 'Untitled')}\n\n"
        if btype == "child_database":
            return f"\U0001f5c3\ufe0f **Sub-database:** {block['child_database'].get('title', 'Untitled')}\n\n"

        # Link preview
        if btype == "link_preview":
            return f"\U0001f517 [Link]({block['link_preview'].get('url', '')})\n\n"

        # Equation
        if btype == "equation":
            return f"$${block['equation'].get('expression', '')}$$\n\n"

        return ""

    def _handle_image(self, block):
        img = block["image"]
        url = (img.get("file", {}).get("url")
               if img.get("type") == "file"
               else img.get("external", {}).get("url"))

        if not url:
            return ""

        ext = get_ext_from_url(url)
        idx = len(self.images) + 1

        try:
            data = http_download(url)
            b64 = base64.b64encode(data).decode("ascii")
        except Exception as e:
            self.errors.append(f"Failed to download image {idx}: {e}")
            log.error(f"  Failed to download image {idx}: {e}")
            return f"![Image download failed](assets/image_{idx}_failed.{ext})\n\n"

        # Detect actual format from bytes (Notion URLs often lie about format)
        real_media_type = detect_media_type(data)
        if real_media_type:
            ext_map = {"image/png": "png", "image/jpeg": "jpg",
                       "image/gif": "gif", "image/webp": "webp"}
            ext = ext_map.get(real_media_type, ext)

        # Skip AI naming for tiny images (<100x100) and unsupported formats (SVG)
        dims = get_image_dimensions(data)
        skip_ai = not real_media_type  # SVG or unknown format
        if dims and dims[0] < 100 and dims[1] < 100:
            skip_ai = True

        if skip_ai:
            fallback = f"{idx}-{slugify(self.current_heading, 30)}.{ext}"
            filename = fallback
        else:
            filename = self.ai.name_image(
                b64, ext, idx, self.current_heading, self.last_paragraph,
                media_type=real_media_type
            )
        self.images.append({"filename": filename, "base64": b64})

        caption = rich_text_to_md(img.get("caption"))
        if not caption:
            caption = filename.rsplit(".", 1)[0].replace("-", " ")

        # Constrain wide images to max 300px width (height scales proportionally)
        dims = get_image_dimensions(data)
        if dims and dims[0] > 300:
            return (f'<p align="center"><a href="assets/{filename}" target="_blank">'
                    f'<img src="assets/{filename}" alt="{caption}" '
                    f'width="300" /></a></p>\n\n')
        return (f'<p align="center"><a href="assets/{filename}" target="_blank">'
                f'<img src="assets/{filename}" alt="{caption}" /></a></p>\n\n')

    def _handle_file(self, block, btype):
        fd = block[btype]
        url = (fd.get("file", {}).get("url")
               if fd.get("type") == "file"
               else fd.get("external", {}).get("url"))
        name = fd.get("name", f"attachment_{len(self.attachments)}")

        if not url:
            return ""

        try:
            data = http_download(url)
            b64 = base64.b64encode(data).decode("ascii")
            self.attachments.append({"filename": name, "base64": b64})
            return f"\U0001f4ce [{name}](attachments/{name})\n\n"
        except Exception as e:
            self.errors.append(f"Failed to download {name}: {e}")
            log.error(f"  Failed to download attachment {name}: {e}")
            return ""

    def _handle_table(self, block):
        children = block.get("children", [])
        if not children:
            return ""

        md = ""
        for i, row in enumerate(children):
            if row.get("type") != "table_row":
                continue
            cells = row["table_row"]["cells"]
            row_str = " | ".join(rich_text_to_md(cell) for cell in cells)
            md += f"| {row_str} |\n"
            if i == 0:
                md += f"| {' | '.join('---' for _ in cells)} |\n"
        return md + "\n"


# ============================================================
# GIT CLIENT
# ============================================================
class GitClient:
    def __init__(self, base_url, token, owner):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.owner = owner
        self.headers = {
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _encode_path(self, file_path: str) -> str:
        """URL-encode each path segment (handles spaces and special chars)."""
        return "/".join(urllib.parse.quote(seg, safe="") for seg in file_path.split("/"))

    def flush_directory(self, repo, dir_path):
        """Delete all files in a directory (non-recursive, one level).
        Used to clean stale assets/attachments before re-committing."""
        encoded = self._encode_path(dir_path)
        url = f"{self.base_url}/repos/{self.owner}/{repo}/contents/{encoded}"
        time.sleep(0.15)

        try:
            listing = http_request(url, headers=self.headers, allow_404=True)
        except Exception as e:
            log.debug(f"  Flush: could not list {dir_path}: {e}")
            return 0

        if not listing or not isinstance(listing, list):
            return 0

        deleted = 0
        for entry in listing:
            if entry.get("type") != "file":
                continue
            sha = entry.get("sha")
            path = entry.get("path")
            if not sha or not path:
                continue
            try:
                encoded_path = self._encode_path(path)
                delete_url = f"{self.base_url}/repos/{self.owner}/{repo}/contents/{encoded_path}"
                time.sleep(0.15)
                http_request(
                    delete_url, method="DELETE", headers=self.headers,
                    body={"sha": sha, "message": f"Flush stale file: {path}"}
                )
                deleted += 1
            except Exception as e:
                log.warning(f"  Flush: failed to delete {path}: {e}")

        if deleted:
            log.info(f"  Flushed {deleted} files from {dir_path}")
        return deleted

    def upsert_file(self, repo, file_path, content_b64, message):
        """Create or update a file in a Gitea repo."""
        encoded = self._encode_path(file_path)
        url = f"{self.base_url}/repos/{self.owner}/{repo}/contents/{encoded}"
        time.sleep(0.15)

        sha = None
        try:
            existing = http_request(url, headers=self.headers, allow_404=True)
            if existing:
                sha = existing.get("sha")
        except Exception:
            pass

        body = {"content": content_b64, "message": message}
        if sha:
            body["sha"] = sha

        method = "PUT" if sha else "POST"
        http_request(url, method=method, headers=self.headers, body=body)
        return True


# ============================================================
# DISCORD NOTIFICATION
# ============================================================
def send_discord(webhook_url, message, level, has_errors):
    """Send Discord notification based on DISCORD_LEVEL.
    level: 0=never, 1=errors only, 2=always
    """
    if not webhook_url or level == 0:
        return
    if level == 1 and not has_errors:
        return

    try:
        if len(message) > 1950:
            message = message[:1950] + "\n... (truncated)"
        http_request(
            webhook_url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "NotionGitBackup/1.0",
            },
            body={"content": message}
        )
        log.info("Discord notification sent")
    except Exception as e:
        log.warning(f"Discord notification failed: {e}")


# ============================================================
# MAIN SYNC LOGIC
# ============================================================
def sync_page(notion, git, ai, page_id, title, dir_name,
              repo, dry_run=False):
    converter = MarkdownConverter(ai, page_title=title)

    log.info(f"  Fetching blocks for: {title}")
    blocks = notion.get_block_children(page_id)

    body_md = converter.convert(blocks)
    now = datetime.now(timezone.utc).isoformat()

    front_matter = "\n".join([
        "---",
        f'title: "{title}"',
        "source: notion",
        f"notion_id: {page_id}",
        f"backed_up: {now}",
        "---",
        ""
    ])
    markdown = f"{front_matter}\n# {title}\n\n{body_md}"

    files_committed = 0
    commit_errors = []

    if dry_run:
        log.info(f"  [DRY RUN] Would flush: {dir_name}/assets/, {dir_name}/attachments/")
        log.info(f"  [DRY RUN] Would commit: {dir_name}/index.md "
                 f"({len(markdown)} chars, {len(converter.images)} images, "
                 f"{len(converter.attachments)} attachments)")
        return {
            "title": title,
            "files": 1 + len(converter.images) + len(converter.attachments),
            "errors": converter.errors,
        }

    # Flush stale assets and attachments before re-committing
    git.flush_directory(repo, f"{dir_name}/assets")
    git.flush_directory(repo, f"{dir_name}/attachments")

    # Commit markdown
    try:
        md_b64 = base64.b64encode(markdown.encode("utf-8")).decode("ascii")
        date_str = now.split("T")[0]
        git.upsert_file(repo, f"{dir_name}/index.md", md_b64,
                        f"Backup: {title} ({date_str})")
        files_committed += 1
    except Exception as e:
        commit_errors.append(f"index.md: {e}")
        log.error(f"  Failed to commit markdown: {e}")

    # Commit images
    for img in converter.images:
        try:
            git.upsert_file(repo, f"{dir_name}/assets/{img['filename']}",
                            img["base64"],
                            f"Backup image: {img['filename']}")
            files_committed += 1
        except Exception as e:
            commit_errors.append(f"{img['filename']}: {e}")
            log.error(f"  Failed to commit image {img['filename']}: {e}")

    # Commit attachments
    for att in converter.attachments:
        try:
            git.upsert_file(
                repo, f"{dir_name}/attachments/{att['filename']}",
                att["base64"],
                f"Backup attachment: {att['filename']}")
            files_committed += 1
        except Exception as e:
            commit_errors.append(f"{att['filename']}: {e}")
            log.error(f"  Failed to commit attachment: {e}")

    log.info(f"  \u2713 {title}: {files_committed} files committed "
             f"({len(converter.images)} images, "
             f"{len(converter.attachments)} attachments)")

    return {
        "title": title,
        "files": files_committed,
        "errors": converter.errors + commit_errors,
        "images": len(converter.images),
        "attachments": len(converter.attachments),
        "ai_failures": ai.failures,
    }


def main():
    env_path = None
    full_sync = False
    dry_run = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--full-sync":
            full_sync = True
        elif args[i] == "--dry-run":
            dry_run = True
        elif args[i].startswith("--env="):
            env_path = args[i].split("=", 1)[1]
        elif args[i] == "--env" and i + 1 < len(args):
            i += 1
            env_path = args[i]
        elif args[i] in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        i += 1

    # Load configuration from .env
    env = load_env(env_path)
    config = build_config(env)
    state = load_state()

    # Initialize clients
    notion = NotionClient(config["notion_token"])
    git = GitClient(config["git_base_url"], config["git_token"],
                    config["git_owner"])
    ai = AiNamer(
        enabled=config["ai_enabled"],
        api_url=config["ai_api_url"],
        api_key=config.get("ai_api_key", ""),
        model=config.get("ai_model", "claude-haiku-4-5-20251001"),
    )

    discord_level = config["discord_level"]
    discord_level_labels = {0: "disabled", 1: "errors only", 2: "always"}

    log.info("=" * 60)
    log.info("Notion \u2192 Git Backup")
    log.info(f"Databases: {len(config['databases'])}")
    if config["ai_enabled"] and config.get("ai_api_key"):
        log.info(f"AI image naming: enabled (model: {config['ai_model']}, key: ...{config['ai_api_key'][-6:]})")
    else:
        log.info(f"AI image naming: disabled")
    log.info(f"Discord notifications: {discord_level_labels.get(discord_level, 'unknown')}")
    if full_sync:
        log.info("Mode: FULL SYNC (ignoring last sync times)")
    if dry_run:
        log.info("Mode: DRY RUN (no commits)")
    log.info("=" * 60)

    summary = {}
    tz_name = config.get("timezone", "UTC")
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
        now = datetime.now(tz).strftime("%d %b %Y, %H:%M %Z")
    else:
        now = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")

    for db in config["databases"]:
        db_id = db["notion_database_id"]
        repo = db["git_repo"]
        label = db["label"]

        if full_sync:
            synced_pages = {}
        else:
            synced_pages = state.get("synced_pages", {})

        log.info(f"\n\U0001f4c2 {label}")
        log.info(f"   Repo: {repo}")

        try:
            pages = notion.query_database(db_id)
        except Exception as e:
            log.error(f"   Failed to query database: {e}")
            summary[label] = {"pages": [], "files": 0, "errors": [str(e)]}
            continue

        log.info(f"   Total pages in database: {len(pages)}")

        # First pass: build slug map for ALL pages to detect collisions
        slug_counts = {}
        for page in pages:
            title = extract_title(page)
            slug = slugify(title, 80)
            if not slug:
                slug = page["id"][:8]
            slug_counts[slug] = slug_counts.get(slug, 0) + 1

        changed = []
        for page in pages:
            page_id = page["id"]
            edited_str = page["last_edited_time"]
            edited = datetime.fromisoformat(edited_str.replace("Z", "+00:00"))

            stored = synced_pages.get(page_id, "1970-01-01T00:00:00.000Z")
            page_last_sync = datetime.fromisoformat(
                stored.replace("Z", "+00:00"))

            title = extract_title(page)
            dir_name = slugify(title, 80)
            if not dir_name:
                dir_name = page["id"][:8]

            # Append short page ID on slug collision
            if slug_counts.get(dir_name, 0) > 1:
                dir_name = f"{dir_name}-{page_id[:8]}"

            if edited > page_last_sync:
                changed.append({
                    "id": page_id,
                    "title": title,
                    "dir_name": dir_name,
                    "last_edited": edited_str,
                })

        if not changed:
            log.info("   \u2713 No changes since last sync")
            summary[label] = {"pages": [], "files": 0, "errors": []}
            continue

        log.info(f"   Pages to sync: {len(changed)}")

        db_summary = {"pages": [], "files": 0, "errors": []}

        for j, page_info in enumerate(changed, 1):
            log.info(f"\n   [{j}/{len(changed)}] {page_info['title']}")
            try:
                result = sync_page(
                    notion, git, ai,
                    page_info["id"], page_info["title"],
                    page_info["dir_name"], repo, dry_run
                )
                db_summary["pages"].append(result["title"])
                db_summary["files"] += result["files"]
                db_summary["errors"].extend(result.get("errors", []))

                # Save per-page sync state so interrupted runs
                # resume from where they left off
                if not dry_run:
                    state.setdefault("synced_pages", {})[page_info["id"]] = page_info["last_edited"]
                    save_state(state)

            except Exception as e:
                log.error(f"   \u2717 Failed: {e}")
                db_summary["errors"].append(f"{page_info['title']}: {e}")

        summary[label] = db_summary

    # ── Discord summary ──
    log.info("\n" + "=" * 60)

    total_pages = sum(len(s["pages"]) for s in summary.values())
    total_files = sum(s["files"] for s in summary.values())
    total_errors = sum(len(s["errors"]) for s in summary.values())

    hostname = socket.gethostname()

    if total_pages == 0:
        discord_msg = ("\U0001f4cb **Notion \u2192 Git Backup**\n\n"
                       "\u2705 No changes detected. All pages up to date.\n"
                       f"\U0001f5a5\ufe0f Host: `{hostname}`")
        log.info("No changes detected across all databases.")
    else:
        discord_msg = "\U0001f4cb **Notion \u2192 Git Backup Complete**\n\n"
        for label, s in summary.items():
            if not s["pages"]:
                continue
            icon = "\u26a0\ufe0f" if s["errors"] else "\u2705"
            discord_msg += f"{icon} **{label}**\n"
            discord_msg += f"  Pages synced: {len(s['pages'])}\n"
            discord_msg += f"  Files committed: {s['files']}\n"
            pages_list = s["pages"]
            if len(pages_list) <= 10:
                discord_msg += f"  Pages: {', '.join(pages_list)}\n"
            else:
                discord_msg += (
                    f"  Pages: {', '.join(pages_list[:10])} "
                    f"(+{len(pages_list) - 10} more)\n")
            if s["errors"]:
                discord_msg += f"  \u274c Errors: {len(s['errors'])}\n"
            discord_msg += "\n"
        discord_msg += f"\U0001f550 Sync completed at {now}\n"
        discord_msg += f"\U0001f5a5\ufe0f Host: `{hostname}`"

        log.info(f"Summary: {total_pages} pages, {total_files} files, "
                 f"{total_errors} errors")

    if not dry_run:
        send_discord(config.get("discord_webhook_url", ""), discord_msg,
                     discord_level, total_errors > 0)

    if total_errors > 0:
        log.warning(f"Completed with {total_errors} errors")
        sys.exit(1)
    else:
        log.info("Completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("\nInterrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
