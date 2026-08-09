"""The single scan cache in detect.py and the `age_seconds` it reports.

The counter below replaces `detect._scan_sessions` — the function that actually
walks /proc — not the `scan_sessions`/`scan_snapshot` façade in front of it. That
is the whole point: these tests must prove a real scan was avoided (or forced),
which a counter on the façade cannot show while a cache hides behind it.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import detect
import main

# The cache is a module global; conftest's autouse `_reset_scan_cache` clears it
# between tests, so nothing here needs to.


def _counting_scan(monkeypatch: pytest.MonkeyPatch, delay: float = 0.0) -> list[int]:
    """Replace the real /proc walk with a counter; returns the (1-element) count."""
    calls = [0]
    lock = threading.Lock()

    def fake_scan(*a: Any, **k: Any) -> list[dict[str, Any]]:
        with lock:
            calls[0] += 1
        if delay:
            time.sleep(delay)
        return [{"pid": 1, "state": "idle"}]

    monkeypatch.setattr(detect, "_scan_sessions", fake_scan)
    return calls


def _fake_clock(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    """Drive detect's clock through `ticks`, then hold the last value.

    Swaps the module reference detect holds rather than time.monotonic itself:
    patching the stdlib globally would hand this fake clock to every thread in the
    process. `time.time` is frozen at the epoch so an implementation reading the
    wall clock instead of the monotonic one reports a wildly wrong age.
    """
    pending = list(ticks)

    def monotonic() -> float:
        return pending.pop(0) if len(pending) > 1 else pending[0]

    monkeypatch.setattr(detect, "time", SimpleNamespace(monotonic=monotonic, time=lambda: 0.0))


async def test_cache_hit_scans_once(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # APP_SCAN_CACHE_TTL=1 → two calls within the second share one /proc walk.
    monkeypatch.setattr(main.settings, "scan_cache_ttl", 1.0)
    calls = _counting_scan(monkeypatch)
    assert (await client.get("/api/sessions")).status_code == 200
    assert (await client.get("/api/sessions")).status_code == 200
    assert calls[0] == 1


async def test_expired_cache_rescans(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.settings, "scan_cache_ttl", 1.0)
    calls = _counting_scan(monkeypatch)
    _fake_clock(monkeypatch, 100.0, 101.5)  # second call lands past the TTL
    await client.get("/api/sessions")
    await client.get("/api/sessions")
    assert calls[0] == 2


@pytest.mark.parametrize("ttl", [0.0, -1.0])
async def test_cache_disabled_forces_a_real_scan(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, ttl: float
) -> None:
    """0 disables the cache; a negative TTL means the same, never "forever".

    This is the criterion a second cache layer would silently break: with one
    stacked above this one, `ttl=0` would disable only the outer copy and the
    inner one would still serve rows up to its own TTL old. Counting the real
    walks is what makes the promise checkable.
    """
    monkeypatch.setattr(main.settings, "scan_cache_ttl", ttl)
    calls = _counting_scan(monkeypatch)
    await client.get("/api/sessions")
    await client.get("/api/sessions")
    assert calls[0] == 2


async def test_ttl_zero_is_not_shadowed_by_a_default_ttl(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # detect's own default TTL is 1.0s. A request asking for ttl=0 must still get a
    # fresh walk — i.e. the module default must not leak in as a floor.
    assert detect._SCAN_TTL > 0
    monkeypatch.setattr(main.settings, "scan_cache_ttl", 0.0)
    calls = _counting_scan(monkeypatch)
    _fake_clock(monkeypatch, 100.0, 100.1)  # well inside detect's own 1s default
    await client.get("/api/sessions")
    body = (await client.get("/api/sessions")).json()
    assert calls[0] == 2
    assert body["age_seconds"] == 0.0


def test_concurrent_callers_scan_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # scan_snapshot is called from a threadpool: concurrent callers are real
    # threads, and the lock must collapse them onto a single scan.
    calls = _counting_scan(monkeypatch, delay=0.05)
    start = threading.Barrier(8)
    ages: list[float] = []
    ages_lock = threading.Lock()

    def call() -> None:
        start.wait()
        _, age = detect.scan_snapshot(True, 5.0)
        with ages_lock:
            ages.append(age)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls[0] == 1
    assert len(ages) == 8


def _call_concurrently(n: int, ttl: float) -> tuple[list[float], list[float]]:
    """Fire `n` threads into scan_snapshot at once; return their (ages, latencies)."""
    start = threading.Barrier(n)
    ages: list[float] = []
    latencies: list[float] = []
    results_lock = threading.Lock()

    def call() -> None:
        start.wait()
        t0 = time.perf_counter()
        _, age = detect.scan_snapshot(True, ttl)
        elapsed = time.perf_counter() - t0
        with results_lock:
            ages.append(age)
            latencies.append(elapsed)

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return ages, latencies


# ttl=1.0 is the shipped default, ttl=0.0 the documented "no cache" setting. In
# both the walk is made to outlast the TTL, which is what used to break the
# collapsing: single-flight is orthogonal to reuse-across-time and must hold in
# BOTH. (A walk SHORTER than the TTL was already collapsed by the TTL alone, so it
# would not exercise this at all.)
@pytest.mark.parametrize(("ttl", "scan_s"), [(1.0, 1.15), (0.0, 0.5)])
def test_a_slow_scan_still_collapses_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch, ttl: float, scan_s: float
) -> None:
    """N callers arriving during one walk cost ONE walk, not N serialised ones.

    The cache is stamped with the instant the walk STARTED, so a walk slower than
    the TTL produces an entry that is already expired when it is published. Before
    the fix every queued caller therefore missed and re-walked *under the lock*:
    40 pollers against a 1.5 s walk with ttl=1.0 meant 40 walks and 60 s for the
    last one. At ttl=0 the TTL never helped at all and the same pile-up happened
    with any walk duration.
    """
    calls = _counting_scan(monkeypatch, delay=scan_s)
    ages, latencies = _call_concurrently(8, ttl)

    assert calls[0] == 1
    # One walk's worth of waiting for everyone, not one each (8 × scan_s).
    assert max(latencies) < scan_s * 2
    # The waiters are told the truth about what they got: rows whose idle_seconds
    # were measured when the walk started, i.e. one walk-duration ago — even though
    # that exceeds the TTL they asked for.
    assert max(ages) >= scan_s * 0.8
    # INCLUDING the caller that did the walk: it holds the same rows, measured at
    # the same instant, so it must be told the same age. Reporting 0.0 to the
    # walker made one client see the very same data as a walk-duration fresher
    # than every other client polling that instant.
    assert min(ages) >= scan_s * 0.8


def test_sequential_callers_still_rescan_at_ttl_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Single-flight must not decay into a cache: a caller that arrives after the
    # walk was published (no walk in flight to join) gets its own walk at ttl=0.
    calls = _counting_scan(monkeypatch, delay=0.02)
    detect.scan_snapshot(True, 0.0)
    detect.scan_snapshot(True, 0.0)
    assert calls[0] == 2


def test_age_is_measured_from_the_start_of_the_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression guard on the stamping instant: `idle_seconds` inside the rows is
    # read at the top of the walk, so age_seconds must count from there. Stamping
    # the END instead (the tempting way to stop the pile-up above) would report an
    # age optimistic by the walk's whole duration, and every client subtracting it
    # would show its sessions less idle than they are.
    scan_s = 0.20
    _counting_scan(monkeypatch, delay=scan_s)
    detect.scan_snapshot(True, 5.0)
    _, age = detect.scan_snapshot(True, 5.0)
    assert age >= scan_s


def test_cache_is_keyed_on_collect_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    # A caller passing a different flag must never be served the other snapshot.
    calls = _counting_scan(monkeypatch)
    detect.scan_snapshot(True, 5.0)
    detect.scan_snapshot(False, 5.0)
    assert calls[0] == 2


async def test_age_seconds_is_the_walk_duration_when_fresh(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not 0.0 even on a fresh scan: the rows are measured when the walk STARTS, so
    # they are already one walk old when the response is built. With an instant
    # (stubbed) walk that is a few microseconds — the invariant, not the number,
    # is what matters here; the slow-walk case is asserted above.
    monkeypatch.setattr(main.settings, "scan_cache_ttl", 1.0)
    _counting_scan(monkeypatch)
    body = (await client.get("/api/sessions")).json()
    assert 0.0 < body["age_seconds"] < 0.05


async def test_age_seconds_grows_when_cached(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.settings, "scan_cache_ttl", 5.0)
    _counting_scan(monkeypatch)
    _fake_clock(monkeypatch, 100.0, 100.25)
    await client.get("/api/sessions")
    body = (await client.get("/api/sessions")).json()
    assert body["age_seconds"] == pytest.approx(0.25)


def test_age_seconds_is_the_true_age_of_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The age must describe the walk that produced the rows, not a later copy.

    With two caches in series the reported age measured the outer layer only and
    under-reported the real one, so a client subtracting it still showed every
    remote session less idle than it was.
    """
    calls = _counting_scan(monkeypatch)
    # Two clock reads for the walking call (start, then return), one per cache hit.
    _fake_clock(monkeypatch, 100.0, 100.4, 100.9)
    # The walker itself is told 0.4 — the walk started at t=100.0 and returned at
    # t=100.4, so its rows are already 0.4 s old. Reporting 0.0 here was the bug.
    assert detect.scan_snapshot(True, 5.0)[1] == pytest.approx(0.4)
    assert detect.scan_snapshot(True, 5.0)[1] == pytest.approx(0.9)
    assert calls[0] == 1


def test_age_seconds_uses_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # The age comes from time.monotonic(): a wall clock stepped backwards by NTP
    # would otherwise yield a negative duration. Drive a fake monotonic clock and
    # check the reported age follows it (it would not if time.time() were used).
    _counting_scan(monkeypatch)
    _fake_clock(monkeypatch, 100.0, 100.25)

    # Walk starts at 100.0, returns at 100.25 → 0.25; the cache hit that follows
    # reads the held 100.25 and reports the same age for the same rows.
    assert detect.scan_snapshot(True, 5.0)[1] == pytest.approx(0.25)
    assert detect.scan_snapshot(True, 5.0)[1] == pytest.approx(0.25)


async def test_response_fields_unchanged(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The HTML page keeps working untouched: age_seconds is added, nothing removed.
    _counting_scan(monkeypatch)
    body = (await client.get("/api/sessions")).json()
    assert set(body) == {"count", "sessions", "auth_required", "dev", "instance", "age_seconds"}
    assert body["count"] == 1
    assert body["sessions"][0]["pid"] == 1


def test_scan_sessions_facade_still_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other callers (and the TUI/GTK copies of detect.py) use the rows-only façade.
    _counting_scan(monkeypatch)
    rows = detect.scan_sessions()
    assert rows == [{"pid": 1, "state": "idle"}]
