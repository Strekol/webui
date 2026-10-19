# Claude Code Watcher — Web UI

[![ci](https://github.com/claude-watcher/webui/actions/workflows/ci.yml/badge.svg)](https://github.com/claude-watcher/webui/actions/workflows/ci.yml)

A crude, **read-only** web view of the running Claude Code sessions on this
machine, served over HTTP with a single self-contained page. **Display only**: no
terminal focus, no actions, no signalling.

## Endpoints

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/` | open | the HTML page (polls `/api/sessions`) |
| GET | `/healthz` | open | `{"status":"ok"}` |
| GET | `/api/meta` | open | `{"auth_required": bool}` |
| GET | `/api/sessions` | gated | `{count, sessions[], age_seconds, auth_required, dev, instance}` |

`age_seconds` is how old the served `/proc` snapshot is. Never `0.0`, not even on a
fresh scan: the rows are measured when the walk STARTS, so they are already one
walk old when the response is built — and it can exceed `APP_SCAN_CACHE_TTL`
(see below). It is measured with a monotonic
clock, so it never goes backwards. Subtract it when deriving a session's idle
duration from `idle_seconds` — the latter was measured when the scan ran, not when
the response was sent.

`dev`/`instance` are livereload internals (`dev` true only under `APP_RELOAD`; the
page reloads itself when `instance` changes). Each entry in `sessions[]` carries
`project`, `state`, `bg_shell`, `context_pct`, `tool`, `topic`, `idle_seconds`, `agents[]`
(the subagents it spawned — `{pid, name, type, model}` each, `pid` null for an in-process `Agent`-tool subagent) and `daemon` (true for
the `claude daemon`, which is not a real session) among others.
Ephemeral headless `claude` runs (`--no-session-persistence` in print mode, e.g. T3 Code
title generation) have no transcript and are never listed.

`state` is one of:

| `state` | Meaning |
|---|---|
| `waiting` | blocked on you — a permission prompt or the end of a turn |
| `working` | Claude is computing (registry `busy`/`shell`/`compacting`) |
| `idle` | nothing in flight |

A separate boolean, `bg_shell`, says a background shell outlived the turn (a
`!cmd`, a backgrounded Bash, a Monitor). It is a detail ABOUT the session, not a
state: the page renders it as a `⚙ sh` marker beside the badge, and it changes
neither the ordering nor the idle duration — Claude handed control back.

Rows are sorted **waiting → working → idle**, then by project.
Only availability ranks here, so a background shell never steals the
spotlight from a session that actually needs you. The TUI and GTK clients apply
the identical rule — the same session must render the same way whether it is read
locally or over this API.

## Auth (optional)

Empty `APP_AUTH_TOKEN` → the API is open. Set it → `/api/sessions` requires the
token, accepted three ways — **all headers**:

- `X-API-Key: <token>`
- `Authorization: Bearer <token>`
- `Authorization: Basic <base64>` — so `curl http://remote:<token>@host/api/sessions`
  works. The token is the **password**; when the password is empty the username is
  used instead, which makes `http://<token>@host/` work too. The username is never
  checked (`remote` is a convention, not a requirement).

> **This server speaks plain HTTP.** It terminates no TLS — there is no
> `ssl_certfile` knob — so an `https://` URL pointed straight at it fails with
> `SSL: RECORD_LAYER_FAILURE`. Every example here is `http://` on purpose. Put a
> reverse proxy (nginx, Caddy, Traefik) in front of it if you need TLS, and point
> clients at the proxy's `https://` URL; or use an SSH tunnel to a loopback URL,
> which keeps the token off the wire entirely. The four client READMEs carry the
> same warning — this is the repo that owns the fact, so it states it here.

> **No token in the URL.** A `?key=<token>` query form used to be accepted; it was
> removed. The access log records `query_params` on every request, so the token
> landed in cleartext on stdout — thousands of times over, since the page polls
> continuously — and from there into whatever keeps the logs. A URL-borne secret
> also leaks through browser history, `Referer` headers and proxy logs. The page
> never needed it: it has always sent `X-API-Key`. Putting a token in a query
> string now simply fails with 401 — and note the query is still logged, so do not
> put secrets there.

A rejected request is answered with `WWW-Authenticate: Bearer`, on purpose: a
`Basic` challenge would make every browser loading the page pop a native
credentials dialog. The page probes `/api/meta` and only shows a key prompt when a
token is required.

## Display options

The ⚙ button in the header reveals a filter bar (mirrors the TUI options). Each
choice is stored per-browser in `localStorage` and applied client-side; the API
is unchanged:

| Option | Values | Effect |
|---|---|---|
| sort | `default` / `idle` | `default` keeps the server order (waiting → working → idle, then project); `idle` floats the most-recently-idle sessions to the top of the idle group |
| idle | `none` / `loose` / `precise` | idle duration shown on idle rows: hidden, `[Nd ]HH:MM`, or `[Nd ]HH:MM:SS` |
| topic | on / off | show the AI title / last prompt under each session |
| agents | on / off | show the count of subagents each session spawned (hover for the list) |
| daemons | on / off | show the `claude daemon` row (marked `(D)`); off hides it |
| cards | on / off | roomier spacing between rows |

The bar itself is hidden on every load (not persisted). Idle durations are
computed **server-side** (`idle_seconds` in each session) rather than from the
browser clock.

## Run

```bash
make install                      # uv sync
make run                          # → http://localhost:8000
APP_AUTH_TOKEN=hunter2 make run   # require a key
```

Config is via `APP_*` env vars / `.env` (see `.env.example`): `APP_HOST`,
`APP_PORT`, `APP_LOG_LEVEL`, `APP_JSON_LOGS`, `APP_AUTH_TOKEN`,
`APP_ALLOW_INSECURE_BIND`, `APP_RELOAD`, `APP_SHOW_AGENTS`, `APP_SCAN_CACHE_TTL`. Set
`APP_SHOW_AGENTS=false` to skip process-subagent detection — it spares a
`/proc/<pid>/cmdline` read per non-`claude` process on each scan, worth it on a host
with thousands of processes. In-process subagents are still read: a running one
keeps its session `working`, only the `agents[]` list is emptied.

`APP_SCAN_CACHE_TTL` (seconds, default `1.0`) is how long one `/proc` snapshot is
reused **across time** — pollers arriving within the TTL of the last scan share it.
`0` (or any negative value) turns off reuse *across time*: each request arriving
after the previous scan finished gets a real scan. It does **not** turn off
single-flight — callers that queue while a walk is running still share its result
rather than each starting their own, and are told its real age (which may exceed
the TTL they asked for). Those are two different things, and conflating them is
what made this setting dangerous: at `ttl=0`, 40 concurrent pollers used to cost
40 `/proc` walks.

Requests that arrive **while a scan is running** share its result at any TTL,
including `0` — one scan per burst, never one per caller, however slow the scan or
short the TTL. Those callers are served rows whose `age_seconds` is the scan's real
age (it can exceed the TTL: it is how long the walk took). There is exactly one
cache (in `detect.py`), so `age_seconds` is the true age of the rows rather than of
some copy of them.

### Bind safety

Binds **`127.0.0.1`** by default. Session data (cwd paths, AI topics, project
names) is **unauthenticated unless `APP_AUTH_TOKEN` is set**, so binding a
non-loopback `APP_HOST` (e.g. `0.0.0.0`) with no token is **refused at startup** —
set a token, or pass `APP_ALLOW_INSECURE_BIND=true` to opt in.

## Container

Detection reads the host `/proc` (so `--pid=host`) and resolves each session's
transcript from **absolute host paths** — `~/.claude`, plus any custom
`CLAUDE_CONFIG_DIR` profile (e.g. `~/.claude-work`) baked into the session's env.
To see every profile, expose the **host home at the same path**, run as the **host
uid** (the `~/.claude*` dirs are `0700`), and set `HOME` to the host home so the
default-profile lookup resolves there too:

```bash
make docker-build

# All profiles (default + every ~/.claude-*):
docker run --rm --pid=host -u $(id -u):$(id -g) -e HOME=$HOME \
  -v $HOME:$HOME:ro -p 8000:8000 \
  claude-watcher-webui:local        # == `make docker-run`

# Default ~/.claude profile only — narrower mount:
docker run --rm --pid=host -u $(id -u):$(id -g) -e HOME=$HOME \
  -v $HOME/.claude:$HOME/.claude:ro -p 8000:8000 \
  claude-watcher-webui:local
```

The image bakes `APP_HOST=0.0.0.0` (so `-p` works) and `APP_ALLOW_INSECURE_BIND=true`
(the netns is the boundary; `-p` is your exposure choice). **Auth is still off by
default** — once you publish the port off a trusted network, set `APP_AUTH_TOKEN`.

## Detection

`detect.py` is a self-contained copy of the watcher detection logic, trimmed to
read-only. It reads `/proc`, the session registry under `~/.claude/sessions/` and
the transcript `.jsonl` files to derive each session's state, context % and topic;
if the upstream heuristics change, align it by hand.

## Develop

```bash
make test     # uv run pytest
make lint     # pre-commit run --all-files
```
