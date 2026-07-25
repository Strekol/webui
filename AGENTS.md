# AGENTS.md — claude-watcher-webui

## What this is

A crude, **read-only** web view of the Claude Code sessions running on this
machine. FastAPI serves a JSON endpoint and one self-contained HTML page that
polls it. **Display only** — no terminal focus, no actions, no signalling.

## Layout

| File | Role |
|---|---|
| `detect.py` | Session detection backend, trimmed to read-only (no X11/window/kitty/focus/kill). Reads `/proc`, `~/.claude/sessions/<pid>.json` (registry) and the transcript `.jsonl`. One `scan_proc()` walk finds both sessions and their subagents (comm≠`claude`, spotted by `--agent-id`/`--parent-session-id`), attached to each session's `agents[]`. The daemon (`claude daemon run …`) is emitted as a minimal `daemon:true` row (`state:"daemon"`); the page marks it `(D)` and can hide it. Holds **the** scan cache (`_scan_cache` + `_scan_seq` behind `_SCAN_LOCK`, monotonic-stamped, keyed on `collect_agents`): `scan_snapshot(collect_agents, ttl)` returns `(rows, age)`, `scan_sessions()` is the rows-only façade. Two mechanisms, do not conflate them — `ttl` bounds reuse across time, `_scan_seq` collapses callers that queue during a walk onto its result and keeps working at `ttl=0`. The age is the rows' true age for every caller, walker included. Do not add a second cache above it (rationale in the code, above `_SCAN_LOCK`). |
| `auth.py` | Optional shared-secret gate, HEADERS ONLY (`X-API-Key` / `Bearer` / `Basic`). The `?key=` query form was removed: the access-log middleware records `query_params`, so the token was written in cleartext on every poll. Open when `APP_AUTH_TOKEN` is empty. In a `Basic` payload the token is the password, or the username when the password is empty (`http://remote:<tok>@host/` and `http://<tok>@host/` both work — `http`, this server terminates no TLS); the username is never validated. The 401 challenge stays `WWW-Authenticate: Bearer` — `Basic` would make browsers pop a native credentials dialog. |
| `main.py` | FastAPI app: `/healthz`, `/api/sessions` (guarded), `/api/meta` (open), `/` (the page). Holds no cache of its own — it calls `detect.scan_snapshot(show_agents, APP_SCAN_CACHE_TTL)` and forwards the age it returns. |
| `static/index.html` | Single polling page, no build step, no external deps. |

`detect.py` is a self-contained copy of the watcher detection logic — align it by
hand if the upstream heuristics change.

## Endpoints

- `GET /healthz` → `{"status": "ok"}`
- `GET /api/meta` → `{"auth_required": bool}` (always open; UI uses it to decide whether to prompt)
- `GET /api/sessions` → `{count, sessions[], age_seconds, auth_required, dev, instance}` (guarded when a token is set; `dev`/`instance` are livereload internals). Each session carries `agents[]` (`{pid, name, type, model}` per spawned subagent). `age_seconds` is the age of the served snapshot (never `0.0` — the rows are measured when the walk starts; can exceed `APP_SCAN_CACHE_TTL`; monotonic-derived so never negative) — a client must subtract it from `idle_seconds`, which was measured at scan time.

## Run

```bash
make install        # uv sync
make run            # uv run python main.py  → http://localhost:8000
APP_AUTH_TOKEN=hunter2 make run   # require a key
make test
make lint
```

Container: detection reads the **host** `/proc` and `~/.claude`, so run with
`--pid=host -v $HOME/.claude:/app/.claude:ro` (see `make docker-run`).

Binds **`127.0.0.1` by default**. Session data (cwd paths, topics, project names)
is unauthenticated unless `APP_AUTH_TOKEN` is set, so binding a non-loopback
`APP_HOST` with no token is **refused at startup** — set a token, or pass
`APP_ALLOW_INSECURE_BIND=true` to opt in (the container does, behind `-p`).

## Conventions

uv for deps; ruff (line length 110) + mypy strict, both enforced by pre-commit
(`make lint`). Modern type hints only (`list[str]`, `str | None`). Health endpoint
at `/healthz`. Structured logging via `fastapi-structured-logging`. Dockerfile is
multi-stage (build with uv, run the `.venv` binary).
