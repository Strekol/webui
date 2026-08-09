import base64
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import Request

from auth import make_auth_dependency

# Sentinel row the stubbed scan returns: an authorised response carrying it proves
# the stub is the seam the route actually uses (a real /proc walk would return the
# host's own sessions — or none at all — but never this).
_STUB_ROW = {"pid": 1, "state": "idle"}


@pytest.fixture
async def secured_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """A client whose app instance enforces a fixed token."""
    import importlib

    import main as main_mod

    monkeypatch.setenv("APP_AUTH_TOKEN", "s3cret")
    main = importlib.reload(main_mod)
    # Patch the real /proc walk, NOT the scan_sessions() façade: /api/sessions goes
    # through detect.scan_snapshot(), so a stub on the façade is never reached and
    # this suite silently walks /proc once per authorised request.
    monkeypatch.setattr(main.detect, "_scan_sessions", lambda *a, **k: [_STUB_ROW])

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    # Drop the env var BEFORE reloading: monkeypatch only restores it after this
    # fixture's teardown, so reloading first would rebuild a *guarded* app and leak
    # the token into every test module that runs later.
    monkeypatch.delenv("APP_AUTH_TOKEN", raising=False)
    importlib.reload(main_mod)  # restore unguarded app for other tests


async def test_rejects_without_token(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions")
    assert r.status_code == 401


async def test_accepts_api_key_header(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200
    # Also pins the stub seam: without it this suite walks the real /proc.
    assert r.json()["sessions"] == [_STUB_ROW]


async def test_accepts_bearer(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


async def test_rejects_the_token_in_the_query_string(secured_client: httpx.AsyncClient) -> None:
    """`?key=` was an accepted form and is now REFUSED, on purpose.

    The access-log middleware records `query_params` on every request, so the
    token landed in cleartext on stdout — thousands of times, since the page
    polls continuously. Headers only; the page always used `X-API-Key` anyway.
    """
    r = await secured_client.get("/api/sessions?key=s3cret")
    assert r.status_code == 401


def _basic(user: str, password: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


async def test_accepts_basic_token_as_password(secured_client: httpx.AsyncClient) -> None:
    # `curl https://remote:<token>@host/api/sessions` — the documented remote syntax.
    r = await secured_client.get("/api/sessions", headers=_basic("remote", "s3cret"))
    assert r.status_code == 200


async def test_accepts_basic_any_username(secured_client: httpx.AsyncClient) -> None:
    # The username is a convention, not a credential: it is never validated.
    r = await secured_client.get("/api/sessions", headers=_basic("whoever", "s3cret"))
    assert r.status_code == 200


async def test_accepts_basic_token_as_username(secured_client: httpx.AsyncClient) -> None:
    # `https://s3cret@host/` → base64("s3cret:"): empty password falls back to the user.
    r = await secured_client.get("/api/sessions", headers=_basic("s3cret", ""))
    assert r.status_code == 200


async def test_basic_password_wins_over_username(secured_client: httpx.AsyncClient) -> None:
    # Both filled → the password is the token; a token-shaped username is ignored.
    assert (await secured_client.get("/api/sessions", headers=_basic("s3cret", "nope"))).status_code == 401
    assert (await secured_client.get("/api/sessions", headers=_basic("nope", "s3cret"))).status_code == 200


async def test_rejects_basic_both_fields_empty(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions", headers=_basic("", ""))
    assert r.status_code == 401


async def test_rejects_basic_wrong_password(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions", headers=_basic("remote", "nope"))
    assert r.status_code == 401


@pytest.mark.parametrize("payload", ["not-base64!!", "", "cmVtb3Rl", "//w="])
async def test_rejects_malformed_basic(secured_client: httpx.AsyncClient, payload: str) -> None:
    # Bad base64, empty payload, a payload with no colon, and valid base64 that is
    # not UTF-8 (b"\xff\xfc") must all 401 rather than raise a 500.
    r = await secured_client.get("/api/sessions", headers={"Authorization": f"Basic {payload}"})
    assert r.status_code == 401


async def test_basic_challenge_stays_bearer(secured_client: httpx.AsyncClient) -> None:
    # Accepting Basic must NOT turn the challenge into `Basic`: every browser
    # loading the page would then pop a native credentials dialog.
    r = await secured_client.get("/api/sessions")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


async def test_meta_unauthenticated(secured_client: httpx.AsyncClient) -> None:
    # /api/meta is always open so the UI can decide whether to prompt.
    r = await secured_client.get("/api/meta")
    assert r.status_code == 200
    assert r.json() == {"auth_required": True}


async def test_wrong_token_rejected(secured_client: httpx.AsyncClient) -> None:
    r = await secured_client.get("/api/sessions", headers={"X-API-Key": "nope"})
    assert r.status_code == 401


async def test_open_dependency_is_noop() -> None:
    dep = make_auth_dependency(None)
    # No token, but an open gate must not raise.
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}
    await dep(Request(scope))  # must not raise


def _req_with_token_header(token: str) -> Request:  # latin-1 : limite d’un en-tête HTTP
    # Query params decode as UTF-8 (unlike HTTP headers, which are latin-1), so a
    # non-ASCII token survives this channel intact.
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/sessions",
            "headers": [(b"x-api-key", token.encode("latin-1"))],
            "query_string": b"",
        }
    )


async def test_latin1_token_matches() -> None:
    # secrets.compare_digest on str raises TypeError for non-ASCII; we compare
    # bytes, so an accented token authenticates instead of 500-ing. Latin-1 is the
    # limit of what an HTTP header can carry — beyond it, startup refuses (below).
    dep = make_auth_dependency("clé-sücrée")
    await dep(_req_with_token_header("clé-sücrée"))  # must not raise


async def test_non_ascii_token_mismatch_is_401() -> None:
    from fastapi import HTTPException

    dep = make_auth_dependency("clé-sücrée")
    with pytest.raises(HTTPException) as exc:
        await dep(_req_with_token_header("wrong"))
    assert exc.value.status_code == 401
