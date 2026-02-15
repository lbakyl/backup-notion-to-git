# Notion → Git Backup

A single Python script that backs up your Notion databases to self-hosted Gitea (or Forgejo) repositories as clean markdown with images and attachments.

**Zero external dependencies** — pure Python 3 standard library. No pip, no Node.js, no Docker required.

## Features

- **Incremental sync** — only processes pages modified since the last run (per-page tracking)
- **30+ Notion block types** — headings, code blocks, tables, callouts, toggles, images, bookmarks, equations, and more
- **Image handling** — downloads images (including from external CDNs), detects actual format from magic bytes, constrains display size, and centers in markdown
- **AI image naming** *(optional)* — sends images to Claude Haiku for descriptive filenames instead of generic numbered names
- **Attachment support** — downloads and commits PDFs, JSON files, and other attachments alongside the markdown
- **Discord notifications** *(optional)* — configurable summary after each run (always, errors only, or never)
- **Slug collision handling** — pages with identical titles get unique directory names via page ID suffixes
- **URL-safe file paths** — handles spaces and special characters in filenames
- **Dry run mode** — preview what would sync without committing anything
- **Resumable** — saves state after each page, so interrupted runs pick up where they left off

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/notion-gitea-backup.git
cd notion-gitea-backup

# 2. Configure
cp .env.example .env
nano .env    # Fill in your values

# 3. Test
python3 notion-gitea-backup.py --dry-run

# 4. Run
python3 notion-gitea-backup.py
```

## Prerequisites

- **Python 3.8+** (already on most Linux systems)
- A **Notion integration** with access to your databases
- A **Gitea/Forgejo** instance with API access
- *(Optional)* An Anthropic API key for AI image naming
- *(Optional)* A Discord webhook for notifications

## Setup

### 1. Create a Notion integration

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → **New integration**
2. Copy the token (starts with `ntn_`)
3. **⚠️ Critical**: For each database you want to back up, click **⋯ → Connections** and add your integration. Without this, the API silently returns empty results.

### 2. Find your Notion database IDs

Open each database as a full page in Notion. The URL looks like:

```
https://www.notion.so/your-workspace/DATABASE_ID?v=...
```

Copy the 32-character hex string (without dashes).

### 3. Create Gitea repositories

Create one repository per database. Initialize each with a README. Generate an API token under **Settings → Applications** with `repo` scope.

### 4. Configure `.env`

Copy `.env.example` to `.env` and fill in your values. The file is self-documented with comments explaining each setting.

Key settings:

| Variable | Required | Description |
|---|---|---|
| `NOTION_TOKEN` | Yes | Your Notion integration token |
| `GIT_BASE_URL` | Yes | Gitea API URL (e.g. `https://gitea.example.com/api/v1`) |
| `GIT_TOKEN` | Yes | Gitea API token with `repo` scope |
| `GIT_OWNER` | Yes | Gitea username or org that owns the repos |
| `AI_NAMING_ENABLED` | No | `0` to disable, `1` to enable AI image naming |
| `DISCORD_LEVEL` | No | `0` = never, `1` = errors only, `2` = always |
| `DB1_NOTION_ID` | Yes | Notion database ID (32-char hex) |
| `DB1_GIT_REPO` | Yes | Target Gitea repository name |
| `DB1_LABEL` | Yes | Human-readable label for logs and Discord |

Add more databases as `DB2_*`, `DB3_*`, etc. No upper limit.

## Usage

```bash
# Normal incremental run
python3 notion-gitea-backup.py

# Force full re-sync (ignore last sync times)
python3 notion-gitea-backup.py --full-sync

# Preview without committing
python3 notion-gitea-backup.py --dry-run

# Use a custom .env location
python3 notion-gitea-backup.py --env /path/to/.env
```

## Scheduling with cron

```bash
crontab -e
```

```cron
# Notion backup every day at 7 AM
0 7 * * * /usr/bin/python3 /opt/notion-gitea-backup/notion-gitea-backup.py >> /var/log/notion-gitea-backup.log 2>&1
```

Verify your Python path with `which python3`.

### Prepare Log file + Log rotation

```bash
sudo touch /var/log/notion-gitea-backup.log
chown your_username:your_username /var/log/notion-gitea-backup.log

sudo nano /etc/logrotate.d/notion-gitea-backup
```

```
/var/log/notion-gitea-backup.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
```

## How it works

```
cron / manual run
  └── notion-gitea-backup.py
        ├── Read .env configuration
        ├── Read sync-state.json (per-page sync timestamps)
        ├── For each configured database:
        │     ├── Query Notion API for all pages
        │     ├── Filter to pages changed since last sync
        │     ├── Detect slug collisions, append page ID suffix if needed
        │     └── For each changed page:
        │           ├── Fetch all blocks recursively (handles nested content)
        │           ├── Convert to markdown (30+ block types)
        │           ├── Download images → detect format → optional AI naming
        │           ├── Download attachments (PDFs, JSON, etc.)
        │           ├── Flush these foldesr: /assets/, /directories/
        │           ├── Commit index.md + assets + attachments to Gitea
        │           └── Save sync state (resumable on interrupt)
        ├── Send a Discord summary if allowed in config (with hostname)
        └── Exit 0 (success) or 1 (errors occurred)
```

## Repository structure

Each backed-up page gets its own directory:

```
notion-it-webdev-kb/
├── article-1/
│   ├── index.md
│   ├── assets/
│   │   ├── 1-image1.png
│   │   ├── 2-image2.png
│   │   └── 3-image3.png
│   └── attachments/
│       └── example.yml
├── article2/
│   ├── index.md
│   └── assets/
│       └── ...
└── article3/
    └── index.md
```

## AI image naming

With `AI_NAMING_ENABLED=1` and a valid API key, each image is sent to the configured AI model for a descriptive filename. For example:

- **Without AI**: `1-prerequisites.png`, `2-install-docker.png`
- **With AI**: `1-prerequisites-proxmox-cluster-dashboard.png`, `2-install-docker-terminal-output.png`

The AI naming includes a circuit breaker — after 5 consecutive failures (e.g. API overload), it falls back to standard naming for the rest of the run.

Works with any Messages API-compatible endpoint (Anthropic, OpenRouter, etc.) via the `AI_API_URL` setting.

**Cost**: ~$0.001–0.002 per image with Claude Haiku. This setting is optional.

## Discord notifications

When configured (optional), the script sends a summary to Discord after each run:

```
📋 Notion → Git Backup Complete

✅ How-tos
  Pages synced: 5
  Files committed: 47
  Pages: Article1, Article2, ...

⚠️ Important Documents
  Pages synced: 2
  Files committed: 8
  ❌ Errors: 1

🕐 Sync completed at 2026-02-14T07:00:00Z
🖥️ Host: my-host
```

## Supported Notion block types

Paragraphs, Heading 1/2/3, Bulleted lists, Numbered lists, To-do lists, Code blocks (with language), Images (uploaded + external), File/PDF attachments, Dividers, Quotes, Callouts, Toggles (as `<details>`), Tables, Bookmarks, Embeds, Videos, Column layouts (flattened), Synced blocks, Child pages/databases (as links), Link previews, Equations (LaTeX).

## Files

| File | Description |
|---|---|
| `notion-gitea-backup.py` | The backup script (~1200 lines, zero dependencies) |
| `.env.example` | Configuration template |
| `.env` | Your configuration (not committed) |
| `sync-state.json` | Per-page sync timestamps (auto-generated, not committed) |
| `.gitignore` | Excludes `.env` and `sync-state.json` |

## Troubleshooting

**API silently returns empty results**: You forgot to share the database with your Notion integration. Click **⋯ → Connections** on each database.

**403 errors on image downloads**: Some CDNs block requests without a User-Agent header. The script includes one, but particularly aggressive CDNs may still block. Check the error log for specific URLs.

**Discord notification not sending**: Ensure your webhook URL is correct and `DISCORD_LEVEL` is set to `1` or `2`.

**Slug collisions**: If two pages have identical titles, both get a `-{page_id}` suffix in their directory name. This is handled automatically.

## Disclaimer

This script was designed and architected by a human. The content of individual functions, this readme file as well as testing to simulate various scenarios were completed with the support of Claude Opus 4.6. All parts of the code have been reviewed by a human. While the code is believed to be safe and ready to use, the responsibility lies with the user.

## License

MIT
