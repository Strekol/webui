"""Optional shared-secret gate.

When `APP_AUTH_TOKEN` is unset/empty the API is fully open. When set, every
guarded request must present the token in a HEADER — one of three (crude on
purpose, display-only data):

  - ``X-API-Key: <token>``            (API clients)
  - ``Authorization: Bearer <token>`` (API clients)
  - ``Authorization: Basic <b64>``    (``http://remote:<token>@host/`` URLs)

A `?key=<token>` query form used to be accepted and was REMOVED, deliberately.
The access-log middleware records `query_params` on every request, so the token
landed in cleartext on stdout — and from there in whatever keeps the logs,
forever — thousands of times over, since the page polls continuously (measured:
`query_params='key=sup3rs3cr3t'`). A URL-borne credential also leaks through
browser history, `Referer` headers, proxy logs and shoulder-surfing. Redacting
the log would have patched one of those paths; not putting the secret in the URL
closes all of them. The page never needed it — it has always sent `X-API-Key`
(see `static/index.html`, `authHeaders()`).

Comparison is constant-time to avoid leaking the token length/prefix via timing.
"""

import base64
import secrets
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status


def _basic_token(payload: str) -> str | None:
    """Token carried by a Basic payload: the password, or the username if empty.

    `https://remote:<token>@host/` is the URL shape the watchers use for a remote,
    and curl, requests and every browser turn it into `Basic base64("remote:<token>")`
    — so this server has to mean the same thing, or the syntax is a trap that only
    works with our own client. `https://<token>@host/` becomes `base64("<token>:")`,
    hence the fallback to the username. The username itself is never validated: it
    is not a secret, and rejecting it would only 401 someone who typed their login.
    """
    try:
        user, _, password = base64.b64decode(payload, validate=True).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        # Malformed base64 or non-UTF-8 credentials: no token presented.
        return None
    # Both fields empty → nothing was presented; None means "reject" downstream.
    return password or user or None


def _present_token(request: Request) -> str | None:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if auth.lower().startswith("basic "):
        return _basic_token(auth[6:].strip())
    # Headers only. The query string is NOT consulted — see the module docstring:
    # a credential in the URL is logged, historised and proxied. `request` carries
    # one, we simply never read it.
    return None


def make_auth_dependency(expected: str | None) -> Callable[[Request], Awaitable[None]]:
    """Build a FastAPI dependency enforcing `expected` (no-op when falsy)."""

    # Compare on UTF-8 bytes: secrets.compare_digest on str raises TypeError for
    # any non-ASCII char, which would turn a valid (accented) token into a 500 and
    # silently break auth. Bytes compare is constant-time and accent-safe.
    expected_b = expected.encode() if expected else None

    async def dependency(request: Request) -> None:
        if not expected_b:
            return
        token = _present_token(request)
        if token is None or not secrets.compare_digest(token.encode(), expected_b):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing credentials",
                # Stays `Bearer` even though Basic is accepted: challenging with
                # `Basic` makes every browser loading the page pop a native
                # credentials dialog, which the key prompt in the UI replaces.
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency
