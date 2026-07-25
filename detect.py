"""Claude Code session detection — display-only backend.

Lifted from the GTK/TUI watcher (single source of truth for the detection
heuristics) and trimmed to a read-only subset: no terminal focus, no window
discovery, no process signalling. We only read /proc, the first-party session
registry (~/.claude/sessions/<pid>.json) and the transcript (.jsonl) to derive
each session's state, context %, current tool and topic.
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_CLK_TCK = os.sysconf("SC_CLK_TCK")

_SESSIONS_DIR = Path.home() / ".claude" / "sessions"
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Registry `status` (~/.claude/sessions/<pid>.json) → displayed state.
# 'shell'/'compacting' = the session is working; 'waiting' = blocked (permission).
_STATUS_MAP = {
    "busy": "working",
    "shell": "working",
    "compacting": "working",
    "waiting": "waiting",
    "idle": "idle",
}


def _argv_value(argv: list[str], flag: str) -> str | None:
    """Value following `flag` in a NUL-split cmdline, else None.

    Empty value → None (`or None`). Flag absent or in last position → None.
    """
    try:
        return argv[argv.index(flag) + 1] or None
    except ValueError, IndexError:
        return None


def scan_proc(
    collect_agents: bool = True, proc_root: Path = Path("/proc")
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Single /proc pass → ('claude' sessions, subagents grouped by parent session id).

    One walk feeds both consumers instead of two. An interactive session runs
    comm=='claude'; a launched subagent (Task/swarm) runs the versioned binary
    (comm != 'claude', invisible to the comm filter) and is spotted by its exact
    argv tokens `--agent-id` / `--parent-session-id` — exact token match on the
    NUL-split argv, so a substring buried in a larger argument can't false-positive.

    `collect_agents=False` skips subagent detection entirely: no cmdline read for
    non-'claude' processes → zero overhead (the reason it is togglable — with it on,
    EVERY non-claude process's cmdline is read each scan). comm is read FIRST: a
    cmdline read failure never loses a claude session. `proc_root` is injectable for
    tests.

    Security: the agent fields (name/type/model/pid) are UNTRUSTED process argv, not
    an authenticated feed — any local process can forge --agent-id/--parent-session-id
    to appear as a subagent of another session. Display only; never used for a
    decision.
    """
    try:
        uptime = float((proc_root / "uptime").read_text().split()[0])
    except Exception:
        return [], {}
    procs: list[dict[str, Any]] = []
    agents: dict[str, list[dict[str, Any]]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm is truncated to 15 chars (TASK_COMM_LEN); 'claude' fits. Read as
            # bytes + decode(errors='ignore'): a non-UTF-8 comm (a name set via prctl
            # by any process) would raise UnicodeDecodeError with read_text() — NOT
            # an OSError → the scan would crash every tick.
            comm = (entry / "comm").read_bytes().decode(errors="ignore").strip()
        except OSError:
            continue
        if comm == "claude":
            try:
                stat = (entry / "stat").read_text()
                fields = stat[stat.rindex(")") + 2 :].split()
                starttime = int(fields[19])
                elapsed = int(uptime - starttime / _CLK_TCK)
                start_unix = time.time() - elapsed
            except Exception:
                continue
            # cmdline is read only to tell the daemon (`claude daemon run …`) from an
            # interactive session; unreadable (racing an exec/exit) → treated as a
            # non-daemon, so a session is never lost over it.
            try:
                argv = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\x00")
            except OSError:
                argv = []
            procs.append(
                {
                    "pid": int(entry.name),
                    "elapsed": elapsed,
                    "start_unix": start_unix,
                    "starttime": starttime,
                    "is_daemon": len(argv) > 1 and argv[1] == "daemon",
                }
            )
            continue
        if not collect_agents:
            continue
        # Subagent: comm != 'claude', so cmdline must be read to spot it.
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\x00")
        except OSError:
            continue
        if "--agent-id" not in argv:
            continue
        parent = _argv_value(argv, "--parent-session-id")
        if not parent:
            continue
        # --agent-name may be missing (anonymous agents): fall back to the local part
        # of the id (<name>@<team>).
        name = _argv_value(argv, "--agent-name") or (_argv_value(argv, "--agent-id") or "?").split("@", 1)[0]
        model = (_argv_value(argv, "--model") or "").removeprefix("claude-")
        agents.setdefault(parent, []).append(
            {
                "pid": int(entry.name),
                "name": name,
                "type": _argv_value(argv, "--agent-type"),
                "model": model or None,
            }
        )
    for lst in agents.values():
        lst.sort(key=lambda a: a["name"])
    return procs, agents


def get_cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return None


def get_env(pid: int) -> dict[str, str]:
    """Read /proc/<pid>/environ → dict. Never raises."""
    try:
        return dict(
            kv.split("=", 1)
            for kv in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\x00")
            if "=" in kv
        )
    except Exception:
        return {}


_WORKTREE_MARKER = "/.claude/worktrees/"


def split_worktree(cwd: str | None) -> tuple[str | None, str | None]:
    """Split a Claude worktree cwd into (project root, worktree name).

    <project>/.claude/worktrees/<name>[/subdir] → (<project>, <name>).
    Outside a worktree → (cwd, None). Single source of the worktree marker.
    """
    if cwd and _WORKTREE_MARKER in cwd:
        root, _, rest = cwd.partition(_WORKTREE_MARKER)
        return root, rest.split("/", 1)[0]
    return cwd, None


def cwd_to_project_dir(cwd: str | None, config_dir: str | None = None) -> Path | None:
    if not cwd:
        return None
    # Custom CLAUDE_CONFIG_DIR instance → its JSONLs live under <config_dir>/projects,
    # not ~/.claude/projects. Otherwise state/context get read from the wrong place.
    base = Path(config_dir) / "projects" if config_dir else CLAUDE_PROJECTS_DIR
    # Claude worktree: the transcript is stored under the PARENT project's slug,
    # not the worktree cwd. Fall back to the project root. Harmless outside a
    # worktree; at worst the dir does not exist → None.
    root, _ = split_worktree(cwd)
    # Claude slugifies the cwd by replacing EVERY non-alphanumeric char with '-'
    # (not just '/'), so 'geoffrey.laurent' → 'geoffrey-laurent'.
    slug = re.sub(r"[^a-zA-Z0-9]", "-", root or cwd)
    path = base / slug
    return path if path.exists() else None


DEFAULT_CONTEXT_WINDOW = 1_000_000

# Models that start at 200k — two sub-cases, same starting assumption:
#   - hard 200k: no 1M window exists for that model;
#   - gated: Opus 4.6 / Sonnet 4.6 only reach 1M through Claude Code's
#     "extended context", which depends on the plan (Opus: included on
#     Max/Team/Enterprise, usage credits on Pro; Sonnet 4.6: credits on every
#     plan) and can be switched off with CLAUDE_CODE_DISABLE_1M_CONTEXT=1.
# Matched as substrings of `message.model`, which may be dated
# (`claude-opus-4-5-20251101`), platform-prefixed (`anthropic.claude-opus-4-5…`)
# or a bare alias (`opus`, `haiku`).
CONTEXT_200K = (
    "haiku",  # every Haiku generation, incl. haiku-4-5
    "opus-4-5",
    "opus-4-1",
    "opus-4-2025",  # claude-opus-4-20250514
    "sonnet-4-5",
    "sonnet-4-2025",  # claude-sonnet-4-20250514
    "claude-3",
    "claude-2",
    "opus-4-6",  # gated
    "sonnet-4-6",  # gated
)


def context_window_for(model: str | None, observed_tokens: int = 0) -> int:
    """Context window (tokens) inferred from the model name and observed usage.

    The JSONL records no window size, and nothing distinguishes a gated model
    running at 200k from the same model upgraded to 1M: Claude Code's `[1m]`
    selector never reaches `message.model` (a `claude-opus-5[1m]` session logs
    `claude-opus-5`). So: 1M by default — every model outside CONTEXT_200K is
    1M on every plan — 200k for CONTEXT_200K, then promoted back to 1M as soon
    as one message is observed above 200k, which only a real 1M window allows.

    Assuming 200k first keeps the error on the safe side: an over-reported ctx%
    warns early, an under-reported one hides a session about to compact.
    """
    m = (model or "").lower()
    if any(tag in m for tag in CONTEXT_200K):
        return 1_000_000 if observed_tokens > 200_000 else 200_000
    return DEFAULT_CONTEXT_WINDOW


# Cache {path: (mtime, result)} — avoids re-reading an unchanged JSONL from one
# scan to the next. The hot-path tail almost always holds the state and the last
# assistant usage in the final few KB (bottom-up parse + early break).
_JSONL_CACHE: dict[str, tuple[float, tuple[str | None, int | None, str | None]]] = {}
_JSONL_TAIL_BYTES = 65536


def _read_tail_lines(path: Path, max_bytes: int) -> tuple[list[str], bool]:
    """Last `max_bytes` of the file, split into lines. The bool says whether the
    whole file was read (complete tail → no fallback needed)."""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read()
    lines = data.decode(errors="ignore").split("\n")
    if start > 0 and len(lines) > 1:
        lines = lines[1:]  # potentially truncated first line → dropped
    return lines, start == 0


# Session topic: `ai-title` (aiTitle, generated by Claude) is written once early
# in the JSONL then rarely regenerated; `last-prompt` (lastPrompt) is appended
# every turn. The state tail-read does not see them (title lives outside the
# last few KB). Dedicated cache {path: (last_complete_line_offset, title,
# lastPrompt)}: full scan on first pass, then only the appended delta is re-read.
_TOPIC_CACHE: dict[str, tuple[int, str | None, str | None]] = {}


def _read_topic(path: Path) -> tuple[str | None, str | None]:
    """(aiTitle, lastPrompt) from the JSONL, re-reading only appended bytes."""
    try:
        size = path.stat().st_size
    except OSError:
        return None, None
    title = last_prompt = None
    start = 0
    cached = _TOPIC_CACHE.get(str(path))
    if cached:
        prev, title, last_prompt = cached
        if size == prev:
            return title, last_prompt
        if size > prev:
            start = prev  # delta only (start sits on a line boundary)
        else:
            # size < prev → truncated/rotated file → full rescan from 0; start
            # fresh (title may have vanished → do not keep a stale value).
            title = last_prompt = None
    try:
        with path.open("rb") as f:
            f.seek(start)
            data = f.read()
    except OSError:
        return title, last_prompt
    nl = data.rfind(b"\n")
    if nl == -1:  # no complete line in the delta
        return title, last_prompt
    for line in data[: nl + 1].decode(errors="ignore").split("\n"):
        if '"ai-title"' not in line and '"last-prompt"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "ai-title" and ev.get("aiTitle"):
            title = ev["aiTitle"]
        elif ev.get("type") == "last-prompt" and ev.get("lastPrompt"):
            last_prompt = ev["lastPrompt"]
    if len(_TOPIC_CACHE) > 200:
        _TOPIC_CACHE.clear()
    _TOPIC_CACHE[str(path)] = (start + nl + 1, title, last_prompt)
    return title, last_prompt


def _parse_session_lines(lines: list[str]) -> tuple[str | None, int | None, str | None]:
    """Bottom-up parse: (state, context_pct, tool).

    `tool` = name of the latest assistant message's last tool_use (the current
    tool); `state` is only used as a fallback (registry absent).
    """
    state: str | None = None
    context_pct: int | None = None
    tool: str | None = None
    seen_assistant = False
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("isSidechain"):
            continue
        kind = ev.get("type", "")
        if state is None:
            if kind == "assistant":
                # stop_reason discriminates "working" from "waiting": 'tool_use'
                # (a tool was dispatched, result pending) or a still-streaming
                # message (None) means Claude is busy; only a terminal end-of-turn
                # reason means it handed control back and is waiting on the user.
                sr = (ev.get("message") or {}).get("stop_reason")
                state = "working" if sr in (None, "tool_use", "pause_turn") else "waiting"
            elif kind == "user":
                state = "working"
            elif kind == "system":
                state = "idle"
        if kind == "assistant":
            msg = ev.get("message", {})
            if not seen_assistant:
                seen_assistant = True
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool = block.get("name")
                            break
            if context_pct is None:
                usage = msg.get("usage", {})
                if usage:
                    total = (
                        usage.get("input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                    )
                    if total > 0:
                        window = context_window_for(msg.get("model"), total)
                        context_pct = min(100, round(total * 100 / window))
        if state is not None and context_pct is not None:
            break
    return state, context_pct, tool


def get_session_info_from_jsonl(
    cwd: str | None,
    config_dir: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, int | None, str | None, str | None, float | None]:
    """State + context % + current tool + topic + JSONL mtime.

    Returns (state, context_pct, tool, topic, mtime). `state` only matters as a
    fallback (registry absent); `topic` = AI title, else last prompt; `mtime` =
    last activity (an "idle since" proxy), None if not found. If `session_id` is
    given, target <session_id>.jsonl directly (exact path, no guessing); else the
    most recent .jsonl in the project. Short-circuited by mtime + tail-only read
    (full re-read when needed).
    """
    project_dir = cwd_to_project_dir(cwd, config_dir)
    if not project_dir:
        return None, None, None, None, None
    latest: Path | None = None
    if session_id:
        cand = project_dir / f"{session_id}.jsonl"
        if cand.is_file():
            latest = cand
    if latest is None:
        jsonl_files = [f for f in project_dir.glob("*.jsonl") if f.is_file()]
        if not jsonl_files:
            return None, None, None, None, None
        try:
            latest, _ = max(
                ((f, f.stat().st_mtime) for f in jsonl_files),
                key=lambda x: x[1],
            )
        except OSError, ValueError:
            return None, None, None, None, None
    try:
        mtime = latest.stat().st_mtime
    except OSError:
        return None, None, None, None, None
    key = str(latest)
    cached = _JSONL_CACHE.get(key)
    if cached and cached[0] == mtime:
        result = cached[1]
    else:
        result = (None, None, None)
        try:
            lines, complete = _read_tail_lines(latest, _JSONL_TAIL_BYTES)
            result = _parse_session_lines(lines)
            # Truncated, incomplete tail (state or pct missing) → full re-read.
            if not complete and (result[0] is None or result[1] is None):
                result = _parse_session_lines(latest.read_text(errors="ignore").split("\n"))
        except Exception:
            pass
        if len(_JSONL_CACHE) > 200:
            _JSONL_CACHE.clear()
        _JSONL_CACHE[key] = (mtime, result)
    title, last_prompt = _read_topic(latest)
    topic = title or last_prompt
    return result[0], result[1], result[2], topic, mtime


def get_session_registry(pid: int, starttime: int, config_dir: str | None = None) -> dict[str, Any] | None:
    """First-party session registry written by Claude: <config>/sessions/<pid>.json.

    Primary state source (real-time `status` field) + `sessionId`/`cwd`. The
    registry lives under the instance's CLAUDE_CONFIG_DIR: a session launched with
    a custom config dir writes to <config_dir>/sessions/, NOT ~/.claude/sessions/.
    Looking in the wrong place makes it invisible and wrongly falls back to the
    JSONL. PID-recycling guard: `procStart` must match the current process's
    `starttime` (field 22 of /proc/<pid>/stat), else the file is stale → ignored.
    Returns the dict, or None if absent/unreadable/stale.
    """
    sessions_dir = (Path(config_dir) / "sessions") if config_dir else _SESSIONS_DIR
    try:
        data: dict[str, Any] = json.loads((sessions_dir / f"{pid}.json").read_text())
    except OSError, ValueError:
        return None
    ps = data.get("procStart")
    if ps is not None:
        try:
            if int(ps) != starttime:
                return None
        except TypeError, ValueError:
            pass
    return data


def get_session_state(
    pid: int,
    cwd: str | None,
    starttime: int = 0,
    config_dir: str | None = None,
) -> tuple[str, int | None, str | None, str | None, float | None, str | None]:
    """Session state. Returns (state, context_pct, tool, topic, last_activity, session_id).

    The registry ~/.claude/sessions/<pid>.json (`status` field) wins when present;
    depending on the Claude Code version it may be absent, in which case state is
    derived from the JSONL. The JSONL always provides context % and current tool.
    The registry's `sessionId`, when present, gives the exact JSONL path (else we
    guess it by slugifying the cwd) and is returned so subagents can be attached to
    their parent by `--parent-session-id`.
    """
    reg = get_session_registry(pid, starttime, config_dir)
    session_id = reg.get("sessionId") if reg else None
    # The transcript slug is computed from the session's STARTUP cwd, which the
    # registry records. The live /proc cwd drifts when the dir is renamed or the
    # user cd's mid-session — slugifying it would point at a project dir that does
    # not exist and silently lose ctx/topic/state. Prefer the registry cwd for
    # transcript resolution; the live cwd stays the displayed label (caller's job).
    transcript_cwd = (reg.get("cwd") if reg else None) or cwd
    jsonl_state, context_pct, tool, topic, last_activity = get_session_info_from_jsonl(
        transcript_cwd, config_dir, session_id
    )
    if reg:
        status = reg.get("status", "")
        state = _STATUS_MAP.get(status, "idle")
        # 'shell' persists while a background shell runs (an interactive `!cmd` or
        # a backgrounded Bash), EVEN after Claude handed control back: the status
        # stays stuck on 'shell' while the session is actually waiting on the user.
        # Cross-check the JSONL — if it shows the turn is over (last assistant in a
        # terminal stop_reason → 'waiting'/'idle'), the shell is just a background
        # residue and the real state is the JSONL's, not 'working'. jsonl_state is
        # None if the JSONL is missing: the condition is then false and we keep the
        # old behavior.
        if status == "shell" and jsonl_state in ("waiting", "idle"):
            state = jsonl_state
        # Idle-since: EXACT instant of the registry's last state change (ms epoch).
        # Preferred over the JSONL mtime, which moves for background writes
        # (summaries, todos) without reflecting real inactivity. Fall back to mtime
        # when the field is absent (older Claude version).
        ts = reg.get("statusUpdatedAt") or reg.get("updatedAt")
        if ts is not None:
            try:
                last_activity = float(ts) / 1000.0
            except TypeError, ValueError:
                pass
    else:
        state = jsonl_state or "idle"
    return state, context_pct, tool, topic, last_activity, session_id


def project_label(cwd: str | None) -> str:
    if not cwd:
        return "?"
    parts = Path(cwd).parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else "?"


def resolve_config_dir(env: dict[str, str]) -> str | None:
    """Absolute CLAUDE_CONFIG_DIR from a session's env, or None.

    Expands a leading `~` (a quoted value is not shell-expanded) and rejects a
    relative path: without the session's cwd it would resolve against the watcher's
    cwd → registry/JSONL read from the wrong place. → default (None).
    """
    config_dir = env.get("CLAUDE_CONFIG_DIR") or None
    if not config_dir:
        return None
    config_dir = os.path.expanduser(config_dir)
    return config_dir if os.path.isabs(config_dir) else None


def display_config_dir(path: str | None) -> str | None:
    """Instance name from CLAUDE_CONFIG_DIR.

    Common case ~/.claude-<name> → just <name>. Otherwise a $HOME path → ~.
    """
    if not path:
        return None
    home = str(Path.home())
    collapsed = "~" + path[len(home) :] if path == home or path.startswith(home + "/") else path
    prefix = "~/.claude-"
    if collapsed.startswith(prefix) and len(collapsed) > len(prefix):
        return collapsed[len(prefix) :]
    return collapsed


# The sync /api/sessions route runs in a threadpool, so scans (and the module
# caches above) can be hit by several threads at once. One lock serialises the
# whole scan: it both guards those plain-dict caches and collapses a burst of
# concurrent polls (many browsers) onto a single /proc walk. A short TTL then
# serves that result to everyone for ~1s — bounding /proc cost regardless of fan-out.
_SCAN_LOCK = threading.Lock()
_SCAN_TTL = 1.0
# (timestamp, collect_agents, result). The flag is part of the key so a caller
# passing a different value never gets served a snapshot built the other way.
_scan_cache: tuple[float, bool, list[dict[str, Any]]] | None = None


def scan_sessions(collect_agents: bool = True) -> list[dict[str, Any]]:
    """Read-only snapshot of every running Claude Code session.

    Display-only: no terminal/window resolution, no focus, no signalling. Sorted
    by state priority (waiting > working > idle), then project name. Thread-safe;
    result is cached for _SCAN_TTL seconds and shared across callers (read-only).
    The cache is keyed on `collect_agents` too, so mixing values stays correct.

    `collect_agents` (APP_SHOW_AGENTS) skips subagent detection when False, sparing
    a cmdline read per non-claude process on busy hosts.
    """
    global _scan_cache
    with _SCAN_LOCK:
        now = time.time()
        if _scan_cache is not None and _scan_cache[1] == collect_agents and now - _scan_cache[0] < _SCAN_TTL:
            return _scan_cache[2]
        result = _scan_sessions(collect_agents)
        _scan_cache = (now, collect_agents, result)
        return result


def _scan_sessions(collect_agents: bool = True) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    # Single clock read for the whole scan → idle durations computed SERVER-side so
    # the page never depends on the viewer's wall clock (it may be a remote browser
    # whose clock is skewed vs this host).
    now = time.time()
    # One /proc walk yields both the sessions and the subagents (keyed by their
    # parent's session id) to attach below.
    procs, agents = scan_proc(collect_agents)
    for p in procs:
        pid = p["pid"]
        cwd = get_cwd(pid)
        env = get_env(pid)
        config_dir = resolve_config_dir(env)

        # The daemon shares comm=='claude' but is not a stateful session (no
        # terminal, no JSONL, no pid-keyed registry). Short-circuit the state
        # resolver and emit a minimal row flagged `daemon` so the page can mark it
        # (D) and render it neutral; hiding it is the client's call.
        if p.get("is_daemon"):
            sessions.append(
                {
                    "pid": pid,
                    "project": project_label(cwd),
                    "worktree": None,
                    "display_cwd": cwd or "?",
                    "cwd": cwd or "?",
                    "topic": None,
                    "state": "daemon",
                    "context_pct": None,
                    "tool": None,
                    "elapsed": p["elapsed"],
                    "idle_seconds": None,
                    "config_dir": display_config_dir(config_dir),
                    "agents": [],
                    "daemon": True,
                }
            )
            continue

        state, context_pct, tool, topic, last_activity, session_id = get_session_state(
            pid, cwd, p["starttime"], config_dir
        )
        # "Confirmed" worktree = marker detected AND transcript resolved
        # (last_activity = mtime of the found JSONL). We then show the REAL project
        # (parent root) + a worktree name. Unconfirmed → raw path, no worktree.
        wt_root, wt_name = split_worktree(cwd)
        confirmed_wt = wt_name is not None and last_activity is not None
        sessions.append(
            {
                "pid": pid,
                "project": project_label(wt_root if confirmed_wt else cwd),
                "worktree": wt_name if confirmed_wt else None,
                "display_cwd": (wt_root if confirmed_wt else cwd) or "?",
                "cwd": cwd or "?",
                "topic": topic,
                "state": state,
                "context_pct": context_pct,
                "tool": tool,
                "elapsed": p["elapsed"],
                # Idle duration in seconds, computed here (not in the browser) to be
                # clock-skew proof. None when the transcript/registry gave no
                # timestamp. Clamped to >=0 so a slightly-ahead statusUpdatedAt never
                # yields a negative.
                "idle_seconds": max(0, int(now - last_activity)) if last_activity is not None else None,
                "config_dir": display_config_dir(config_dir),
                # Subagents (Task/swarm) spawned by this session, matched on the
                # registry sessionId == the child's --parent-session-id.
                "agents": agents.get(session_id, []) if session_id else [],
                "daemon": False,
            }
        )
    sessions.sort(key=lambda s: (s["state"] != "waiting", s["state"] != "working", s["project"].lower()))
    return sessions
