from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

import detect


@pytest.fixture(autouse=True)
def _reset_scan_cache() -> Iterator[None]:
    # detect holds the ONE scan cache (~1s by default); clear it between tests so a
    # fast suite doesn't serve one test's snapshot to the next.
    detect._scan_cache = None
    detect._scan_seq = 0
    yield
    detect._scan_cache = None
    detect._scan_seq = 0


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    import main

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
