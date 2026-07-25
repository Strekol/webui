#!/usr/bin/env python3
"""Claude Code Watcher — Web UI.

Crude, read-only HTTP view of the local Claude Code sessions the GTK/TUI watchers
track. Serves a JSON endpoint and a single self-contained HTML page that polls
it. Display only — no focus, no actions. Optionally gated by a shared secret
(see auth.py).
"""

import os
import secrets
from pathlib import Path
from typing import Any

import fastapi_structured_logging
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

import detect
from auth import make_auth_dependency

_STATIC_DIR = Path(__file__).parent / "static"

# Per-process id. With `APP_RELOAD=true` uvicorn re-imports this module on every
# code/template edit → a fresh id → the dev page sees the change and reloads
# itself (crude livereload). Stable for the life of one process otherwise.
INSTANCE = secrets.token_hex(8)


class Settings(BaseSettings):
    # Loopback by default: the API exposes session data (full cwd paths, AI topics,
    # project names) with no auth unless a token is set. Binding wider is opt-in.
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    json_logs: bool | None = None
    # Empty → API is open. Set → required on /api/* and the page bootstrap.
    auth_token: str = ""
    # Explicit acknowledgement to bind a non-loopback host WITHOUT auth (e.g. a
    # container publishing to a trusted network). Off → such a bind is refused.
    allow_insecure_bind: bool = False
    # Detect the subagents (Task/swarm) each session spawned and show them. On by
    # default; turning it off spares a /proc/<pid>/cmdline read per NON-claude
    # process on every scan — worth it on a host with thousands of processes.
    show_agents: bool = True
    # Seconds a /proc snapshot is reused across callers. 0 (or any negative value)
    # disables the cache. Bounds the scan cost when N browsers/watchers poll a busy
    # host; the response carries `age_seconds` so a client knows how old the data is.
    scan_cache_ttl: float = 1.0
    # Dev only: uvicorn auto-reload on edits + browser livereload on restart.
    reload: bool = False

    model_config = SettingsConfigDict(env_prefix="APP_")


# Addresses that only accept connections from the local host.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def insecure_bind_reason(host: str, auth_token: str, allow_insecure: bool) -> str | None:
    """Why this bind is unsafe, or None if it is fine.

    Serving session data unauthenticated on a non-loopback address exposes cwd
    paths / topics / project names to the whole network. Refuse it unless the
    operator opts in (a token, or APP_ALLOW_INSECURE_BIND).
    """
    if auth_token or allow_insecure:
        return None
    if host in _LOOPBACK_HOSTS:
        return None
    return (
        f"refusing to bind {host} with no auth: session data would be exposed to the "
        "network. Set APP_AUTH_TOKEN=<secret>, or APP_ALLOW_INSECURE_BIND=true to "
        "override, or bind APP_HOST=127.0.0.1."
    )


settings = Settings()
log = fastapi_structured_logging.get_logger()
require_auth = make_auth_dependency(settings.auth_token or None)

app = FastAPI(title="Claude Code Watcher — Web UI", version=os.getenv("VERSION", "v0.0.0"))
app.add_middleware(fastapi_structured_logging.AccessLogMiddleware)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Sync def on purpose: the scan does blocking /proc + file I/O. Declared sync,
# Starlette runs it in a threadpool so it never stalls the event loop (and
# /healthz) when several browsers poll concurrently. The cache and its lock live
# in detect.scan_snapshot() — the one place that knows when the rows were built,
# and therefore the only place that can report their true age.
@app.get("/api/sessions", dependencies=[Depends(require_auth)])
def sessions() -> dict[str, Any]:
    rows, age = detect.scan_snapshot(settings.show_agents, settings.scan_cache_ttl)
    return {
        "count": len(rows),
        "sessions": rows,
        "age_seconds": age,
        "auth_required": bool(settings.auth_token),
        # In dev, the page reloads itself when `instance` changes (server restart).
        "dev": settings.reload,
        "instance": INSTANCE,
    }


@app.get("/api/meta")
async def meta() -> dict[str, Any]:
    """Unauthenticated: lets the UI know whether to prompt for a token."""
    return {"auth_required": bool(settings.auth_token)}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


def main() -> None:
    if settings.json_logs is True:
        fastapi_structured_logging.setup_logging(json_logs=True, log_level=settings.log_level)
    elif settings.json_logs is False:
        fastapi_structured_logging.setup_logging(json_logs=False, log_level=settings.log_level)
    else:
        fastapi_structured_logging.setup_logging(log_level=settings.log_level)

    log.info(
        "startup",
        version=os.getenv("VERSION", "v0.0.0"),
        commit_hash=os.getenv("COMMIT_HASH", "00000000-dirty"),
        build_timestamp=os.getenv("BUILD_TIMESTAMP", "1970-01-01T00:00:00+00:00"),
        project_url=os.getenv("PROJECT_URL", "unknown"),
        host=settings.host,
        port=settings.port,
        auth_required=bool(settings.auth_token),
        show_agents=settings.show_agents,
    )

    # Fail fast on an unsafe bind (non-loopback + no auth) unless explicitly allowed.
    reason = insecure_bind_reason(settings.host, settings.auth_token, settings.allow_insecure_bind)
    if reason:
        log.error("insecure_bind_refused", reason=reason)
        raise SystemExit(reason)
    if not settings.auth_token and settings.allow_insecure_bind and settings.host not in _LOOPBACK_HOSTS:
        log.warning("serving_unauthenticated_on_network", host=settings.host)

    # Reload needs an import string (uvicorn re-imports in a worker); pass the app
    # object otherwise to avoid a needless re-import. Watch .py and the template.
    uvicorn.run(
        "main:app" if settings.reload else app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
        access_log=False,
        reload=settings.reload,
        reload_includes=["*.py", "static/*.html"] if settings.reload else None,
    )


if __name__ == "__main__":
    main()
