import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

import detect
from main import insecure_bind_reason


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_bind_loopback_always_ok(host: str) -> None:
    # Loopback is safe even with no auth and no override.
    assert insecure_bind_reason(host, "", False) is None


def test_bind_nonloopback_no_auth_refused() -> None:
    assert insecure_bind_reason("0.0.0.0", "", False) is not None


def test_bind_nonloopback_with_auth_ok() -> None:
    assert insecure_bind_reason("0.0.0.0", "secret", False) is None


def test_bind_nonloopback_with_override_ok() -> None:
    assert insecure_bind_reason("0.0.0.0", "", True) is None


async def test_health(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_meta_open(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/meta")
    assert r.status_code == 200
    assert r.json() == {"auth_required": False}


async def test_sessions_open(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the real /proc walk, not the façade: the route goes through
    # detect.scan_snapshot(), so a stub on scan_sessions() would never be reached.
    monkeypatch.setattr(detect, "_scan_sessions", lambda *a, **k: [{"pid": 1, "state": "idle"}])
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["sessions"][0]["pid"] == 1


def test_scan_sessions_no_procs(monkeypatch: pytest.MonkeyPatch) -> None:
    # No claude processes → empty list, never raises.
    monkeypatch.setattr(detect, "scan_proc", lambda *a, **k: ([], {}))
    assert detect.scan_sessions() == []


def _one_proc(
    monkeypatch: pytest.MonkeyPatch,
    state_tuple: tuple[str, int | None, str | None, str | None, float | None, str | None],
    agents: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    monkeypatch.setattr(
        detect,
        "scan_proc",
        lambda *a, **k: ([{"pid": 1, "elapsed": 10, "start_unix": 0.0, "starttime": 0}], agents or {}),
    )
    monkeypatch.setattr(detect, "get_cwd", lambda pid: "/tmp/proj")
    monkeypatch.setattr(detect, "get_env", lambda pid: {})
    monkeypatch.setattr(detect, "get_session_state", lambda *a, **k: state_tuple)


def test_idle_seconds_computed_server_side(monkeypatch: pytest.MonkeyPatch) -> None:
    # idle_seconds is derived from last_activity on the server (clock-skew proof),
    # clamped to >=0, and present (>= the elapsed since last_activity).
    _one_proc(monkeypatch, ("idle", 5, None, "topic", time.time() - 120, None))
    rows = detect.scan_sessions()
    assert len(rows) == 1
    assert rows[0]["idle_seconds"] is not None
    assert rows[0]["idle_seconds"] >= 119


def test_idle_seconds_none_without_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    _one_proc(monkeypatch, ("working", 5, "Bash", None, None, None))
    rows = detect.scan_sessions()
    assert rows[0]["idle_seconds"] is None


def test_idle_seconds_clamped_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # last_activity slightly in the future (skewed statusUpdatedAt) → 0, never < 0.
    _one_proc(monkeypatch, ("idle", 5, None, None, time.time() + 30, None))
    rows = detect.scan_sessions()
    assert rows[0]["idle_seconds"] == 0


def test_no_agents_yields_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _one_proc(monkeypatch, ("working", 5, "Task", None, None, "sess-1"))
    rows = detect.scan_sessions()
    assert rows[0]["agents"] == []


def test_agents_attached_by_parent_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Subagents keyed by the child's --parent-session-id are attached to the
    # session whose registry sessionId matches.
    agents = {"sess-1": [{"pid": 42, "name": "explorer", "type": "Explore", "model": "opus-4-8"}]}
    _one_proc(monkeypatch, ("working", 5, "Task", None, None, "sess-1"), agents=agents)
    rows = detect.scan_sessions()
    assert rows[0]["agents"] == agents["sess-1"]


def test_agents_not_attached_when_session_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A subagent parented to a different session id is not attached here.
    agents = {"other": [{"pid": 42, "name": "explorer", "type": None, "model": None}]}
    _one_proc(monkeypatch, ("working", 5, "Task", None, None, "sess-1"), agents=agents)
    rows = detect.scan_sessions()
    assert rows[0]["agents"] == []


def test_daemon_row_is_minimal_and_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A daemon proc yields a minimal row flagged `daemon` with a neutral state,
    # and the state resolver is never consulted for it.
    monkeypatch.setattr(
        detect,
        "scan_proc",
        lambda *a, **k: (
            [{"pid": 2, "elapsed": 5, "start_unix": 0.0, "starttime": 0, "is_daemon": True}],
            {},
        ),
    )
    monkeypatch.setattr(detect, "get_cwd", lambda pid: "/tmp/proj")
    monkeypatch.setattr(detect, "get_env", lambda pid: {})
    rows = detect.scan_sessions()
    assert len(rows) == 1
    r = rows[0]
    assert r["daemon"] is True
    assert r["state"] == "daemon"
    assert r["agents"] == [] and r["tool"] is None and r["context_pct"] is None
    assert r["idle_seconds"] is None


def test_non_daemon_row_flagged_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _one_proc(monkeypatch, ("working", 5, "Bash", None, None, "sess-1"))
    rows = detect.scan_sessions()
    assert rows[0]["daemon"] is False


def test_resolve_config_dir() -> None:
    assert detect.resolve_config_dir({}) is None
    assert detect.resolve_config_dir({"CLAUDE_CONFIG_DIR": "relative/dir"}) is None  # relative → None
    assert detect.resolve_config_dir({"CLAUDE_CONFIG_DIR": "/abs/dir"}) == "/abs/dir"


def _write_proc(root: Path, pid: int, comm: str, cmdline: bytes, *, with_stat: bool = True) -> None:
    d = root / str(pid)
    d.mkdir()
    (d / "comm").write_text(comm + "\n")
    (d / "cmdline").write_bytes(cmdline)
    if with_stat:
        # Fields after ") " start at the process state; index 19 = field 22 = starttime.
        (d / "stat").write_text(f"{pid} ({comm}) S " + " ".join(["0"] * 18) + " 42")


def test_scan_proc_classifies_sessions_daemon_and_agents(tmp_path: Path) -> None:
    root = tmp_path
    (root / "uptime").write_text("1000.0 500.0\n")
    _write_proc(root, 100, "claude", b"/usr/bin/claude\x00")  # interactive session
    _write_proc(root, 200, "claude", b"/usr/bin/claude\x00daemon\x00run\x00")  # daemon
    _write_proc(root, 400, "kworker", b"", with_stat=False)  # kernel thread → ignored
    _write_proc(  # subagent: comm != claude, spotted by argv tokens
        root,
        300,
        "1.2.3",
        b"/usr/bin/claude-1.2.3\x00--agent-id\x00foo@team\x00--parent-session-id\x00sess-1"
        b"\x00--agent-name\x00explorer\x00--agent-type\x00Explore\x00--model\x00claude-opus-4-8\x00",
        with_stat=False,
    )

    procs, agents = detect.scan_proc(proc_root=root)
    by_pid = {p["pid"]: p for p in procs}
    assert set(by_pid) == {100, 200}  # only comm=='claude'; kworker ignored
    assert by_pid[100]["is_daemon"] is False
    assert by_pid[200]["is_daemon"] is True
    assert agents == {"sess-1": [{"pid": 300, "name": "explorer", "type": "Explore", "model": "opus-4-8"}]}

    # collect_agents=False skips subagent detection but still classifies the daemon.
    procs2, agents2 = detect.scan_proc(collect_agents=False, proc_root=root)
    assert agents2 == {}
    assert {p["pid"] for p in procs2} == {100, 200}
    assert {p["pid"]: p["is_daemon"] for p in procs2} == {100: False, 200: True}


def test_scan_cache_keyed_on_collect_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two calls <1s apart with different collect_agents must not share a snapshot:
    # the second (False) must not be served the first's agent-bearing result.
    agents = {"sess-1": [{"pid": 42, "name": "x", "type": None, "model": None}]}

    def fake_scan_proc(collect_agents: bool = True, proc_root: object = None) -> object:
        procs = [{"pid": 1, "elapsed": 10, "start_unix": 0.0, "starttime": 0}]
        return procs, (agents if collect_agents else {})

    monkeypatch.setattr(detect, "scan_proc", fake_scan_proc)
    monkeypatch.setattr(detect, "get_cwd", lambda pid: "/tmp/proj")
    monkeypatch.setattr(detect, "get_env", lambda pid: {})
    monkeypatch.setattr(
        detect, "get_session_state", lambda *a, **k: ("working", 5, "Task", None, None, "sess-1")
    )
    assert detect.scan_sessions(True)[0]["agents"] == agents["sess-1"]
    assert detect.scan_sessions(False)[0]["agents"] == []  # not the cached True snapshot


def test_main_refuses_insecure_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    # main() logs startup (incl. show_agents) then aborts a non-loopback bind with no
    # auth before ever starting uvicorn.
    import main

    monkeypatch.setattr(main.settings, "host", "0.0.0.0")
    monkeypatch.setattr(main.settings, "auth_token", "")
    monkeypatch.setattr(main.settings, "allow_insecure_bind", False)
    with pytest.raises(SystemExit):
        main.main()


def test_scan_proc_no_proc_root() -> None:
    # Missing /proc (no uptime) → empty, never raises.
    assert detect.scan_proc(proc_root=Path("/nonexistent-proc")) == ([], {})


def test_get_session_state_from_registry_and_jsonl(tmp_path: Path) -> None:
    # End-to-end over the real parsers: a config-dir instance routes registry and
    # transcript under tmp_path. Asserts state (from registry), context % + tool
    # (from JSONL), last_activity (from statusUpdatedAt) and the session_id return.

    pid, starttime, cwd = 1234, 42, "/tmp/proj"
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / f"{pid}.json").write_text(
        json.dumps(
            {
                "procStart": starttime,
                "sessionId": "sess-1",
                "status": "busy",
                "cwd": cwd,
                "statusUpdatedAt": 1_700_000_000_000,
            }
        )
    )
    # Claude slugifies the cwd (every non-alphanumeric → '-'): "/tmp/proj" → "-tmp-proj".
    proj = tmp_path / "projects" / "-tmp-proj"
    proj.mkdir(parents=True)
    (proj / "sess-1.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "model": "claude-opus-4-8",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1000, "cache_read_input_tokens": 9000},
                    "content": [{"type": "tool_use", "name": "Bash"}],
                },
            }
        )
        + "\n"
    )

    state, ctx, tool, topic, last_activity, session_id = detect.get_session_state(
        pid, cwd, starttime, config_dir=str(tmp_path)
    )
    assert state == "working"  # registry status 'busy' → working
    assert ctx == 1  # 10_000 / 1_000_000 (opus → 1M window) → 1%
    assert tool == "Bash"
    assert topic is None  # no ai-title / last-prompt line
    assert last_activity == 1_700_000_000.0  # statusUpdatedAt ms → s
    assert session_id == "sess-1"


def test_context_window_for() -> None:
    # 1M on every plan: everything outside the 200k list, incl. unknown/bare aliases.
    for model in ("claude-opus-5", "claude-opus-4-8", "claude-sonnet-5", "opus", "sonnet", None):
        assert detect.context_window_for(model) == 1_000_000, model
    # Hard 200k, dated and platform-prefixed IDs included.
    for model in (
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
        "anthropic.claude-opus-4-5-20251101-v1:0",
    ):
        assert detect.context_window_for(model) == 200_000, model
    # Gated models: 200k until a message proves the 1M upgrade is active.
    for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
        assert detect.context_window_for(model) == 200_000, model
        assert detect.context_window_for(model, 199_000) == 200_000, model
        assert detect.context_window_for(model, 250_000) == 1_000_000, model
    # A hard-200k model can never exceed its window, so promotion never fires for it.
    assert detect.context_window_for("claude-haiku-4-5-20251001", 150_000) == 200_000


def _registry_and_transcript(tmp_path: Path, status: str, last_event: dict[str, object]) -> tuple[int, str]:
    """A config-dir instance holding one session: registry `status` + one JSONL event."""
    pid, starttime, cwd = 4321, 7, "/tmp/bg"
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / f"{pid}.json").write_text(
        json.dumps({"procStart": starttime, "sessionId": "sess-bg", "status": status, "cwd": cwd})
    )
    proj = tmp_path / "projects" / "-tmp-bg"
    proj.mkdir(parents=True)
    (proj / "sess-bg.jsonl").write_text(json.dumps(last_event) + "\n")
    return pid, cwd


# stop_reason 'end_turn' → jsonl_state 'waiting'; a system event → 'idle'; 'tool_use'
# → 'working' (a real turn in flight).
_TURN_OVER = {"type": "assistant", "message": {"stop_reason": "end_turn", "content": []}}
# ANY `type: "system"` line means "turn over" to _parse_session_lines — `subtype`
# is never read. Named for what the parser does, not for a discrimination it makes.
_TURN_OVER_SYSTEM = {"type": "system", "subtype": "post_turn"}
_TURN_RUNNING = {"type": "assistant", "message": {"stop_reason": "tool_use", "content": []}}


@pytest.mark.parametrize(
    ("status", "event", "expected"),
    [
        # A stuck 'shell'/'busy' over a finished turn is background work, not a
        # session that needs the user — the exact rule the TUI and GTK apply.
        ("shell", _TURN_OVER, "background"),
        ("busy", _TURN_OVER, "background"),
        ("busy", _TURN_OVER_SYSTEM, "background"),
        # Live turn → no reconciliation; 'compacting' is excluded on purpose.
        ("busy", _TURN_RUNNING, "working"),
        ("compacting", _TURN_OVER, "working"),
    ],
)
def test_stuck_status_over_a_finished_turn_is_background(
    tmp_path: Path, status: str, event: dict[str, object], expected: str
) -> None:
    # The server used to emit the JSONL state ('waiting'/'idle') for 'shell' only,
    # so the same session read over the API claimed it needed the user while the
    # local clients showed it as background.
    pid, cwd = _registry_and_transcript(tmp_path, status, event)
    state, *_ = detect.get_session_state(pid, cwd, starttime=7, config_dir=str(tmp_path))
    assert state == expected


def test_background_sorts_between_working_and_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Row order must match the clients' priority: waiting > working > background > idle.
    states = ["idle", "background", "waiting", "working"]
    procs = [{"pid": i, "elapsed": 1, "start_unix": 0.0, "starttime": 0} for i in range(len(states))]
    monkeypatch.setattr(detect, "scan_proc", lambda *a, **k: (procs, {}))
    monkeypatch.setattr(detect, "get_cwd", lambda pid: "/tmp/proj")
    monkeypatch.setattr(detect, "get_env", lambda pid: {})
    monkeypatch.setattr(
        detect,
        "get_session_state",
        lambda pid, *a, **k: (states[pid], None, None, None, None, None),
    )
    assert [s["state"] for s in detect.scan_sessions()] == ["waiting", "working", "background", "idle"]


def test_a_renamed_registry_topic_wins_over_the_ai_title(tmp_path: Path) -> None:
    # /rename: the user's own wording is what they will recognise in the list, and
    # both clients already prefer it. The server dropped it, so the same session
    # showed 'my task' locally and the AI title over the API.
    pid, starttime, cwd = 5678, 11, "/tmp/named"
    (tmp_path / "sessions").mkdir()
    proj = tmp_path / "projects" / "-tmp-named"
    proj.mkdir(parents=True)
    (proj / "sess-n.jsonl").write_text(json.dumps({"type": "ai-title", "aiTitle": "AI Title"}) + "\n")

    def _state(**extra: object) -> str | None:
        (tmp_path / "sessions" / f"{pid}.json").write_text(
            json.dumps({"procStart": starttime, "sessionId": "sess-n", "status": "idle", "cwd": cwd, **extra})
        )
        return detect.get_session_state(pid, cwd, starttime, config_dir=str(tmp_path))[3]

    assert _state(name="my task") == "my task"
    # 'derived' = auto-generated from the cwd, redundant → the AI title stays.
    assert _state(name="my task", nameSource="derived") == "AI Title"
    assert _state() == "AI Title"


def test_a_rootless_worktree_is_not_the_projects_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `base / ''` IS `base`: with no guard, a cwd whose project root is empty
    # handed back the PROJECTS DIRECTORY itself — which always exists — as if it
    # were a project. The old fallbacks ('' one side, the worktree cwd here) are
    # both wrong: with no root there is simply no project to point at. The
    # slugified worktree cwd is pre-created so the divergence cannot hide behind
    # a path that happens not to exist.
    projects = tmp_path / "projects"
    (projects / "--claude-worktrees-wt").mkdir(parents=True)
    monkeypatch.setattr(detect, "CLAUDE_PROJECTS_DIR", projects)
    assert detect.cwd_to_project_dir("/.claude/worktrees/wt") is None


def test_get_session_state_stale_registry_ignored(tmp_path: Path) -> None:
    # procStart mismatch (PID reuse) → registry ignored; no JSONL either → idle, no id.

    pid = 1234
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / f"{pid}.json").write_text(
        json.dumps({"procStart": 999, "sessionId": "stale", "status": "busy"})
    )
    state, ctx, tool, topic, last_activity, session_id = detect.get_session_state(
        pid, "/tmp/nope", starttime=42, config_dir=str(tmp_path)
    )
    assert state == "idle"
    assert session_id is None


async def test_sessions_route_forwards_show_agents(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The route must pass settings.show_agents through to the scan verbatim. Patch
    # the real walk: scan_snapshot() forwards the flag to it (and keys its cache on
    # it), so this also proves the flag survives the cache layer.
    import main

    captured: dict[str, bool] = {}

    def fake(collect_agents: bool = True) -> list[object]:
        captured["v"] = collect_agents
        return []

    monkeypatch.setattr(main.settings, "show_agents", False)
    monkeypatch.setattr(detect, "_scan_sessions", fake)
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    assert captured["v"] is False


def test_argv_value() -> None:
    assert detect._argv_value(["--model", "opus"], "--model") == "opus"
    assert detect._argv_value(["--model", ""], "--model") is None  # empty → None
    assert detect._argv_value(["--model"], "--model") is None  # last position → None
    assert detect._argv_value(["x", "y"], "--model") is None  # absent → None


async def test_index_serves_filter_controls(client: httpx.AsyncClient) -> None:
    # Guards the filter-bar regressions: controls must be present and the
    # `.controls[hidden]` rule (which lets the ⚙ actually hide them) must exist.
    r = await client.get("/")
    assert r.status_code == 200
    html = r.text
    for needle in (
        'id="opt-sort"',
        'id="opt-idle"',
        'id="opt-topic"',
        'id="opt-agents"',
        'id="opt-daemons"',
        'id="opt-cards"',
        'id="settings-btn"',
        ".controls[hidden]",
    ):
        assert needle in html, needle


def _run_page_js(snippet: str) -> str:
    """Execute a snippet against the page's own JS, via node. Returns stdout.

    The bucket/sort logic lives only in static/index.html and no Python test can
    reach it — which is exactly how a regression shipped: `bucket()` gained the
    `background` state (idle 2 → 3) while its consumer still tested the literal
    `2`, so idle rows silently stopped being idle-sorted and background rows
    started being. Extracting the two functions and running them under node is the
    cheapest thing that turns that class of drift into a red test.
    """
    page = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()
    # Pull the two functions verbatim out of the page — no second copy to drift.
    start = page.index("const BUCKET =")
    end = page.index("function authHeaders", start)
    src = page[start:end]
    proc = subprocess.run(
        ["node", "-e", src + "\nconst prefs={sort:'idle'};\n" + snippet],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_page_idle_sort_orders_idle_rows_and_spares_background() -> None:
    """`sort=idle` must order IDLE rows by idle time and leave background alone.

    Both halves matter: the mode exists for idle rows, and a background session is
    running — the TUI (claude-watcher-tui.py) and the GTK widget both exclude it
    from this ordering for that reason.
    """
    if not shutil.which("node"):
        # Never a silent skip in CI: an untested layer that reports "passed" is how
        # the regression this test exists for reached main in the first place.
        if os.getenv("CI"):
            pytest.fail("node is required to run the page-JS tests in CI")
        pytest.skip("node not installed")

    out = _run_page_js(
        """
        const rows = [
          {state:'idle',       project:'alpha', idle_seconds:9000},
          {state:'idle',       project:'zeta',  idle_seconds:10},
          {state:'background', project:'alpha', idle_seconds:9000},
          {state:'background', project:'zeta',  idle_seconds:10},
        ];
        const s = sortSessions(rows);
        console.log(s.map(r => r.state + ':' + r.project).join(','));
        """
    )
    order = out.split(",")
    # Idle rows: most-recently-idle first → zeta (10s) before alpha (9000s).
    assert order[2:] == ["idle:zeta", "idle:alpha"], out
    # Background rows: server order kept (alphabetical), NOT idle-sorted.
    assert order[:2] == ["background:alpha", "background:zeta"], out


def test_page_bucket_order_matches_the_server() -> None:
    if not shutil.which("node"):
        if os.getenv("CI"):
            pytest.fail("node is required to run the page-JS tests in CI")
        pytest.skip("node not installed")
    out = _run_page_js(
        "console.log(['waiting','working','background','idle','daemon']"
        ".map(st => bucket({state: st})).join(','))"
    )
    # Same priority as detect.scan_sessions' sort and both native clients.
    assert out == "0,1,2,3,3", out
