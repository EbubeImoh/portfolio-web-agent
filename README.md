---
title: portfolio-web-agent
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---
## portfolio-web-agent

This repo hosts Ebube's personal web chat agent. The `app.py` entry point loads résumé data, builds a system prompt, and serves a Gradio UI backed by Gemini with tool support for collecting leads and logging unanswered questions.

### Setup

```bash
# Install dependencies and create the virtual environment
uv sync

# Activate the environment (optional, for local shells)
source .venv/bin/activate
```

### Environment variables

| Variable                               | Required | Purpose                                                 |
| -------------------------------------- | -------- | ------------------------------------------------------- |
| `GOOGLE_API_KEY`                     | ✅       | Auth for the Gemini-compatible OpenAI client.           |
| `PUSHOVER_USER` / `PUSHOVER_TOKEN` | optional | Enables push notifications when tools fire.             |
| `APP_LOG_LEVEL`                      | optional | Python logging level (default `INFO`).                |
| `MAX_USER_MESSAGE_CHARS`             | optional | Hard cap on incoming message length (default `1000`). |
| `MESSAGE_FORBIDDEN_TERMS`            | optional | Comma-separated keywords that block a message.          |

Store them in `.env` (already loaded via `python-dotenv`).

### Required data files

- `data/summary.txt`
- `data/Profile.pdf`

The app caches their contents and watches for file mtime changes, so editing either file will refresh the next time a user sends a message.

### Running the chat app

```bash
uv run python app.py
```

Startup will fail fast if `GOOGLE_API_KEY` or the data files are missing. If Pushover env vars are absent, the app still launches but logs a warning.

### Tool event log

Every time Gemini calls `record_user_details` or `record_unknown_question`, the payload is appended to `data/tool_events.csv` with a UTC timestamp. Use this file to reconcile lead captures or unanswered questions even if push notifications fail.

### uv / requirements workflows

If you need to work with `requirements.txt`, uv can still help:

```bash
# Install from requirements.txt without touching pyproject
uv pip install -r requirements.txt

# Export locked pyproject deps back to requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

Try to keep both dependency sources aligned if external tools rely on `requirements.txt`.
