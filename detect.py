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
    except (ValueError, IndexError):
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

    Ephemeral headless runs are skipped: exact token `--no-session-persistence` plus a
    print-mode token (`-p`/`--print`/`--output-format[=]`/`--input-format[=]`) — the
    flag is ignored interactively. Unreadable cmdline → still listed.

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
            # cmdline tells the daemon (`claude daemon run …`) and ephemeral runs apart;
            # unreadable (racing an exec/exit) → listed as a non-daemon, so a session is
            # never lost over it.
            try:
                argv = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\x00")
            except OSError:
                argv = []
            # Headless ephemeral run: no transcript to watch (see docstring).
            if "--no-session-persistence" in argv and any(
                a in ("-p", "--print", "--output-format", "--input-format")
                or a.startswith(("--output-format=", "--input-format="))
                for a in argv
            ):
                continue
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


def git_worktree(cwd: str | None) -> tuple[str | None, str | None]:
    """PLAIN git worktree: (main repository root, worktree name).

    Nothing to do with the Claude layout (`<project>/.claude/worktrees/<name>`,
    see `split_worktree`): a `git worktree add` drops its checkout anywhere, so
    that cwd carries no marker in its path. The proof is on disk — a worktree
    root holds a `.git` FILE (not a directory) pointing at
    `<repo>/.git/worktrees/<name>`. A `.git` directory = main checkout, stop
    there. A `gitdir:` without `/worktrees/` = submodule.

    DISPLAY only: Claude stores a plain git worktree's transcript under its OWN
    cwd slug, so `cwd_to_project_dir` must NOT climb to the root here. Outside a
    worktree → (None, None).
    """
    if not cwd:
        return None, None
    start = Path(cwd)
    for d in (start, *start.parents):
        dot = d / ".git"
        try:
            if dot.is_dir():
                return None, None
            if not dot.is_file():
                continue
            head = dot.read_text(errors="ignore")[:4096].strip()
        except OSError:
            return None, None
        if not head.startswith("gitdir:"):
            return None, None
        gitdir = head[len("gitdir:") :].strip().split("\n", 1)[0]
        # `worktree.useRelativePaths` writes a gitdir RELATIVE to the worktree
        # directory; resolving it keeps the repository root usable.
        target = Path(gitdir) if Path(gitdir).is_absolute() else (d / gitdir)
        parts = target.parts
        try:
            i = len(parts) - 1 - parts[::-1].index("worktrees")
        except ValueError:
            return None, None
        # <repo>/.git/worktrees/<name>: the root is what precedes the '.git'.
        if i < 2 or parts[i - 1] != ".git" or i + 1 >= len(parts):
            return None, None
        return str(Path(*parts[: i - 1])), parts[i + 1]
    return None, None


def worktree_of(cwd: str | None, transcript_found: bool) -> tuple[str | None, str | None]:
    """(project to DISPLAY, worktree name) — otherwise (cwd, None).

    Claude worktree: the marker sits in the path, so it may be coincidental; it
    only counts once the transcript resolved under the parent root. Plain git
    worktree: the `.git` file IS the proof, nothing to confirm.
    """
    root, name = split_worktree(cwd)
    if name is not None and transcript_found:
        return root, name
    # Unconfirmed Claude marker: do not stop there. A Claude worktree IS a git
    # worktree, and its `.git` file proves what the path merely suggests.
    root, name = git_worktree(cwd)
    return (root, name) if name else (cwd, None)


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
    # EMPTY root ('/.claude/worktrees/wt'): the slug would be '' and `base / ''`
    # IS `base`, so we would hand back the PROJECTS DIRECTORY itself as if it were
    # a project — and it always exists. No fallback is right here (neither '' nor
    # the worktree cwd, which is not where Claude stores the transcript): with no
    # root, there is no project to point at.
    if not root:
        return None
    # Claude slugifies the cwd by replacing EVERY non-alphanumeric char with '-'
    # (not just '/'), so 'geoffrey.laurent' → 'geoffrey-laurent'.
    slug = re.sub(r"[^a-zA-Z0-9]", "-", root)
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


# `system` event subtypes that mark the END OF A TURN — the only ones proving
# Claude handed control back. The others (`informational`, `api_error`,
# `local_command`, `compact_boundary`, …) happen MID-turn: reading them as a
# finished turn reconciled a working session down to background (registry 'busy'
# cross-checked against a JSONL believed idle).
_TURN_END_SUBTYPES = {"turn_duration", "stop_hook_summary", "away_summary"}


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
            elif kind == "system" and ev.get("subtype") in _TURN_END_SUBTYPES:
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


# A sessionId is used to BUILD paths (a direct path and a glob pattern): with no
# guard, a booby-trapped id in the registry reads outside the projects tree
# ('../../elsewhere/secret') or reopens the neighbour bug by playing the wildcard
# ('*'). Require ONE filename component, no '..' — deliberately not a UUID shape:
# the day Claude changes its id format, too tight a guard would silently strip
# ctx/topic from EVERY row.
_SID_RE = re.compile(r"(?!\.\.?\Z)[A-Za-z0-9._-]+\Z")


# (projects root, session_id) -> JSONL path, when it is not under the cwd slug (a
# session resumed from another directory). Avoids rescanning every project on
# each refresh. The root is part of the key: two CLAUDE_CONFIG_DIRs can carry the
# same sessionId.
_SID_PATH_CACHE: dict[tuple[str, str], Path | None] = {}


def _find_transcript(session_id: str, project_dir: Path) -> Path | None:
    """Session JSONL: under the cwd slug, else scanned across the other projects."""
    if not _SID_RE.match(session_id):
        return None
    cand = project_dir / f"{session_id}.jsonl"
    if cand.is_file():
        return cand
    root = project_dir.parent
    key = (str(root), session_id)
    if key in _SID_PATH_CACHE:
        hit = _SID_PATH_CACHE[key]
        # A memorised absence (None) is safe ONLY because the direct path above is
        # tested on every call — a transcript appearing later under the cwd slug
        # is therefore always seen.
        if hit is None or hit.is_file():
            return hit
    if len(_SID_PATH_CACHE) > 200:
        _SID_PATH_CACHE.clear()
    for p in root.glob(f"*/{session_id}.jsonl"):
        _SID_PATH_CACHE[key] = p
        return p
    _SID_PATH_CACHE[key] = None
    return None


def get_session_info_from_jsonl(
    cwd: str | None,
    config_dir: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, int | None, str | None, str | None, float | None]:
    """State + context % + current tool + topic + JSONL mtime.

    Returns (state, context_pct, tool, topic, mtime). `state` only matters as a
    fallback (registry absent); `topic` = AI title, else last prompt; `mtime` =
    last activity (an "idle since" proxy), None if not found. If `session_id` is
    given, target <session_id>.jsonl (under the cwd slug, else in the project it
    actually lives in) and nothing else; otherwise the most recent .jsonl in the
    project. Short-circuited by mtime + tail-only read
    (full re-read when needed).
    """
    project_dir = cwd_to_project_dir(cwd, config_dir)
    if not project_dir:
        return None, None, None, None, None
    latest: Path | None = None
    if session_id:
        # The transcript does not necessarily live under the cwd slug: a session
        # resumed from another directory (`claude -r <id>`) keeps its original
        # JSONL. Falling back to "most recent .jsonl in the project" here read a
        # NEIGHBOUR session's state and attributed it to this one (status, ctx%,
        # topic). Nothing beats wrong: no fallback once the id is known.
        latest = _find_transcript(session_id, project_dir)
        if latest is None:
            return None, None, None, None, None
    else:
        jsonl_files = [f for f in project_dir.glob("*.jsonl") if f.is_file()]
        if not jsonl_files:
            return None, None, None, None, None
        try:
            latest, _ = max(
                ((f, f.stat().st_mtime) for f in jsonl_files),
                key=lambda x: x[1],
            )
        except (OSError, ValueError):
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
    except (OSError, ValueError):
        return None
    ps = data.get("procStart")
    if ps is not None:
        try:
            if int(ps) != starttime:
                return None
        except (TypeError, ValueError):
            pass
    return data


def get_session_state(
    pid: int,
    cwd: str | None,
    starttime: int = 0,
    config_dir: str | None = None,
    since: float | None = None,
) -> tuple[str, int | None, str | None, str | None, float | None, str | None, bool]:
    """Session state. Returns (state, context_pct, tool, topic, last_activity, session_id,
    bg_shell) — bg_shell says a background shell outlived the turn. The state describes
    ONLY Claude's availability: a background shell is a detail ABOUT the session, not
    what the session IS.

    The registry ~/.claude/sessions/<pid>.json (`status` field) wins when present;
    depending on the Claude Code version it may be absent, in which case state is
    derived from the JSONL. The JSONL always provides context % and current tool.
    The registry's `sessionId`, when present, gives the exact JSONL path (else we
    guess it by slugifying the cwd) and is returned so subagents can be attached to
    their parent by `--parent-session-id`.

    `since` (process start) enables in-process subagents: a running one makes the
    session `working` (see session_agents).
    """
    reg = get_session_registry(pid, starttime, config_dir)
    session_id = reg.get("sessionId") if reg else None
    bg_shell = False
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
        # /rename: a user-chosen name (field `name` with nameSource != 'derived' —
        # 'derived' = auto-generated, redundant with the cwd) wins over the JSONL's
        # AI title as the displayed topic. Mirrors the TUI/GTK clients: the same
        # session read locally and read over this API has to show the same topic,
        # and the user's own wording is the one they will recognise in the list.
        reg_name = reg.get("name")
        if reg_name and reg.get("nameSource") != "derived":
            topic = reg_name
        status = reg.get("status", "")
        state = _STATUS_MAP.get(status, "idle")
        # A 'shell'/'busy' status can stay STUCK after the turn ended, but the two
        # do not mean the same thing: 'shell' = a background shell really is
        # running (low-priority state, waiting > working > background > idle);
        # 'busy' = nothing runs, the registry merely stopped being written
        # (interrupted subagents, a session the user backgrounded). 'compacting'
        # is EXCLUDED: real, brief foreground work. jsonl_state None (no JSONL) or
        # 'working' (turn in flight, subagents included) → nothing reconciled.
        # Mirrors the TUI/GTK clients: the same session read locally and over this
        # API has to render the same state.
        if status in ("shell", "busy") and jsonl_state in ("waiting", "idle"):
            state = jsonl_state
            bg_shell = status == "shell"
            reconciled = True
        else:
            reconciled = False
        # Running in-process subagent: Claude works and resumes on its own when it
        # ends (the registry stays 'busy'). Only a REGISTRY `waiting` (permission)
        # wins; a `waiting` reconciled from the JSONL is just an ended turn.
        if (
            since is not None
            and (state == "idle" or reconciled)
            and session_agents(session_id, transcript_cwd, config_dir, since)
        ):
            state = "working"
        # Idle-since: EXACT instant of the registry's last state change (ms epoch).
        # Preferred over the JSONL mtime, which moves for background writes
        # (summaries, todos) without reflecting real inactivity. Fall back to mtime
        # when the field is absent (older Claude version).
        ts = reg.get("statusUpdatedAt") or reg.get("updatedAt")
        if ts is not None:
            try:
                last_activity = float(ts) / 1000.0
            except (TypeError, ValueError):
                pass
    else:
        state = jsonl_state or "idle"
    return state, context_pct, tool, topic, last_activity, session_id, bg_shell


# IN-PROCESS subagents (Agent tool): nothing to see in /proc, their only trace is
# a separate transcript, <transcript without .jsonl>/subagents/agent-<id>.jsonl,
# next to an agent-<id>.meta.json (type, description, model). The session registry
# says nothing about them: a background agent ends the turn, so the session reads
# `idle` while Claude is still working.
_AGENT_HANDBACK_TOOL = "SubagentHandback"
# What Claude writes when an agent is stopped (TaskStop) or interrupted.
_AGENT_INTERRUPTED = "[Request interrupted by user"
# {subagents dir: {file: (mtime, running)}} — replaced on each call by the files
# present only: bounded by the live agents, never wiped wholesale.
_AGENT_CACHE: dict[str, dict[str, tuple[float, bool]]] = {}


def _agent_running(lines: list[str]) -> bool | None:
    """Bottom-up parse of a subagent transcript: is it still running?

    Finished = last assistant message with a terminal stop_reason (end_turn…), a
    SUCCESSFUL result of its SubagentHandback (report delivered to the caller), or
    the interruption marker (TaskStop). A refused handback (`success: false`) or a
    message injected after it (the agent resumes on a notification from one of its
    own background tasks) → running. None: no message in the lines read.
    """
    handback_id = None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        kind = ev.get("type")
        msg = ev.get("message") or {}
        content = msg.get("content")
        blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
        if kind == "assistant":
            if handback_id is not None:
                return not any(
                    b.get("type") == "tool_use"
                    and b.get("id") == handback_id
                    and b.get("name") == _AGENT_HANDBACK_TOOL
                    for b in blocks
                )
            return msg.get("stop_reason") in (None, "tool_use", "pause_turn")
        if kind == "user" and handback_id is None:
            text = (
                content
                if isinstance(content, str)
                else next((b.get("text") for b in blocks if b.get("type") == "text"), None)
            )
            if isinstance(text, str) and text.startswith(_AGENT_INTERRUPTED):
                return False
            res = ev.get("toolUseResult")
            if not (isinstance(res, dict) and res.get("success") is True):
                return True
            handback_id = next((b.get("tool_use_id") for b in blocks if b.get("type") == "tool_result"), None)
            if handback_id is None:
                return True
    return None


def session_agents(
    session_id: str | None, cwd: str | None, config_dir: str | None, since: float
) -> list[dict[str, Any]]:
    """RUNNING in-process subagents of the session → [{pid, name, type, model}].

    `since` = start of the claude process: an agent transcript untouched since then
    belongs to an earlier life of the session (`claude -r`), killed with it before
    its end was written — it would look like it runs forever. pid None: the agent
    lives inside the session's process.
    """
    project_dir = cwd_to_project_dir(cwd, config_dir)
    if not session_id or project_dir is None:
        return []
    transcript = _find_transcript(session_id, project_dir)
    if transcript is None:
        return []
    sub = str(transcript.with_suffix("") / "subagents")
    try:
        entries = list(os.scandir(sub))
    except OSError:
        return []
    old = _AGENT_CACHE.get(sub, {})
    fresh: dict[str, tuple[float, bool]] = {}
    agents: list[dict[str, Any]] = []
    for e in entries:
        if not (e.name.startswith("agent-") and e.name.endswith(".jsonl")):
            continue
        try:
            mtime = e.stat().st_mtime
        except OSError:
            continue
        if mtime < since:
            continue
        cached = old.get(e.path)
        if cached and cached[0] == mtime:
            running = cached[1]
        else:
            path = Path(e.path)
            try:
                lines, complete = _read_tail_lines(path, _JSONL_TAIL_BYTES)
                r = _agent_running(lines)
                if r is None and not complete:
                    r = _agent_running(path.read_text(errors="ignore").split("\n"))
                # None after a full read: transcript just created, no message yet →
                # the agent is starting.
                running = r is not False
            except OSError:
                running = False
        fresh[e.path] = (mtime, running)
        if not running:
            continue
        try:
            meta = json.loads(Path(e.path.removesuffix(".jsonl") + ".meta.json").read_text())
        except (OSError, ValueError):
            meta = None
        if not isinstance(meta, dict):
            meta = {}
        desc, typ, model = (
            v if isinstance(v, str) and v else None
            for v in (meta.get("description"), meta.get("agentType"), meta.get("model"))
        )
        agents.append(
            {
                "pid": None,
                "name": desc or e.name.removeprefix("agent-").removesuffix(".jsonl"),
                "type": typ,
                "model": (model or "").removeprefix("claude-") or None,
            }
        )
    if len(_AGENT_CACHE) > 200:
        _AGENT_CACHE.clear()
    _AGENT_CACHE[sub] = fresh
    agents.sort(key=lambda a: a["name"])
    return agents


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
# serves that result to everyone — bounding /proc cost regardless of fan-out.
#
# This is the ONLY scan cache. A second one layered above it would be worse than
# redundant: the age reported to clients would measure the outer layer only, so it
# would under-report the true age of the rows by up to the inner TTL, and a caller
# asking for ttl=0 would still be served this cache's copy.
_SCAN_LOCK = threading.Lock()
_SCAN_TTL = 1.0
# (monotonic timestamp, collect_agents, result). Monotonic, NEVER the wall clock:
# the stamp is only ever used to derive a DURATION (the cache age below, and the
# `age_seconds` handed to clients), and time.time() would make it jump — possibly
# negative — on an NTP step. The flag is part of the key so a caller passing a
# different value never gets served a snapshot built the other way.
_scan_cache: tuple[float, bool, list[dict[str, Any]]] | None = None
# Bumped once per published walk. A caller samples it BEFORE queueing on the lock;
# if it has moved by the time the lock is theirs, a walk completed while they
# waited and its rows are theirs to reuse. That is single-flight collapsing, and it
# is a SEPARATE mechanism from the TTL: the TTL reuses a snapshot across time, this
# reuses the one walk a burst of simultaneous callers is already paying for. It
# must therefore hold even at ttl=0 — "no reuse across time" is not "N callers, N
# walks, one at a time under the lock".
_scan_seq = 0


def scan_snapshot(collect_agents: bool = True, ttl: float = _SCAN_TTL) -> tuple[list[dict[str, Any]], float]:
    """Session rows plus the age of the snapshot they come from.

    The age is not cosmetic: `idle_seconds` inside each row is measured when the
    scan runs, so a caller served from cache must subtract this age or every
    session looks up to one TTL less idle than it really is.

    `ttl` <= 0 disables reuse and forces a real scan — and because this is the only
    cache, that promise actually holds. It does NOT disable single-flight: callers
    that queued while a walk was running still share its result (and are told its
    real age, which may exceed `ttl`), because the alternative is N serialised
    walks for one instant's worth of data.
    """
    global _scan_cache, _scan_seq
    # Sampled outside the lock on purpose: everything published between here and
    # the lock acquisition below is, by definition, a walk that finished while this
    # caller was waiting for it.
    seq_on_arrival = _scan_seq
    with _SCAN_LOCK:
        now = time.monotonic()
        if _scan_cache is not None and _scan_cache[1] == collect_agents:
            age = now - _scan_cache[0]
            # Two independent reasons to serve this entry: it is still within the
            # TTL, or it was published while we queued (single-flight).
            if age < ttl or _scan_seq != seq_on_arrival:
                return _scan_cache[2], age
        # The stamp is the instant the walk STARTS, and the entry is published only
        # once the walk has finished. Start-time because `idle_seconds` inside the
        # rows is measured from the same clock read at the top of _scan_sessions(),
        # so `age_seconds` (now - stamp) keeps describing the instant the data
        # describes; stamping the end instead would report an age optimistic by the
        # walk's whole duration. Publishing only at the end is what makes the entry
        # visible to waiters as one completed unit — and the seq bump above it is
        # what lets those waiters take it even when it was born already older than
        # the TTL (a 1.5 s walk under a 1.0 s TTL: before, every queued caller
        # missed and re-walked under this lock, so 40 pollers cost 40 walks and the
        # last one waited 60 s).
        result = _scan_sessions(collect_agents)
        _scan_cache = (now, collect_agents, result)
        _scan_seq += 1
        # NOT 0.0: the rows were measured at `now`, before the walk, so by the time
        # this returns they are already one walk old. Reporting 0.0 to the caller
        # that did the walk — while every waiter gets an exact age — makes the same
        # rows look up to a walk-duration fresher to one client than to another
        # polling the same instant. Negligible on a healthy host, 1.5 s on the busy
        # one this cache exists for.
        return result, time.monotonic() - now


def scan_sessions(collect_agents: bool = True) -> list[dict[str, Any]]:
    """Read-only snapshot of every running Claude Code session.

    Display-only: no terminal/window resolution, no focus, no signalling. Sorted
    by state priority (waiting > working > background > idle), then project name.
    Thread-safe; cached for _SCAN_TTL seconds and shared across callers
    (read-only). Callers that need the snapshot's age want `scan_snapshot()`.
    """
    return scan_snapshot(collect_agents)[0]


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
                    "bg_shell": False,
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

        state, context_pct, tool, topic, last_activity, session_id, bg_shell = get_session_state(
            pid, cwd, p["starttime"], config_dir, since=p["start_unix"]
        )
        # Confirmed Claude worktree (marker detected AND transcript resolved) or
        # plain git worktree (proved by its `.git` file): show the REAL project
        # (repository root) + a worktree name. Otherwise raw path, no worktree.
        wt_root, wt_name = worktree_of(cwd, last_activity is not None)
        sessions.append(
            {
                "pid": pid,
                "project": project_label(wt_root),
                "worktree": wt_name,
                "display_cwd": wt_root or "?",
                "cwd": cwd or "?",
                "topic": topic,
                "state": state,
                "bg_shell": bg_shell,
                "context_pct": context_pct,
                "tool": tool,
                "elapsed": p["elapsed"],
                # Idle duration in seconds, computed here (not in the browser) to be
                # clock-skew proof. None when the transcript/registry gave no
                # timestamp. Clamped to >=0 so a slightly-ahead statusUpdatedAt never
                # yields a negative.
                "idle_seconds": max(0, int(now - last_activity)) if last_activity is not None else None,
                "config_dir": display_config_dir(config_dir),
                # Subagents spawned by this session: process ones (swarm), matched on
                # the registry sessionId == the child's --parent-session-id, then the
                # running in-process ones (Agent tool).
                # The state already counts the in-process ones, even when the list is
                # hidden (collect_agents=False).
                "agents": (
                    (agents.get(session_id, []) if session_id else [])
                    + session_agents(session_id, cwd, config_dir, p["start_unix"])
                )
                if collect_agents
                else [],
                "daemon": False,
            }
        )
    # Only AVAILABILITY ranks here (waiting > working > idle). A background shell
    # had a rank of its own and pushed an available session above idle ones; it is
    # a detail about the session, carried by `bg_shell`, not an ordering criterion.
    sessions.sort(
        key=lambda s: (
            s["state"] != "waiting",
            s["state"] != "working",
            s["project"].lower(),
        )
    )
    return sessions
