import asyncio
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supoclip_mcp import server
from supoclip_mcp.client import SupoClipError, SupoClipTimeoutError
from test_client_download import configured_client


def client(monkeypatch, request):
    monkeypatch.setattr(server, "_client", lambda: SimpleNamespace(request=request))


def test_sleep_never_exceeds_the_wait_deadline(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(server, "asyncio", SimpleNamespace(wait_for=asyncio.wait_for, sleep=sleep, TimeoutError=asyncio.TimeoutError))
    request = AsyncMock(return_value={"status": "processing", "progress": 30})
    client(monkeypatch, request)
    result = json.loads(asyncio.run(server.supoclip_wait_for_task("task", timeout_seconds=5, poll_interval_seconds=60)))
    assert sleeps == [5]
    assert request.await_count == 1
    assert request.await_args.kwargs["timeout_seconds"] == 5
    assert result["timed_out"] is True
    assert result["task"] == {"status": "processing", "progress": 30}


@pytest.mark.parametrize("first_status", [None, {"status": "processing", "progress": 42}])
def test_stalled_request_is_cancelled_and_keeps_last_state(monkeypatch, first_status):
    cancelled = []
    calls = 0

    async def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 and first_status is not None:
            return first_status
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    client(monkeypatch, request)
    started = time.monotonic()
    result = json.loads(asyncio.run(server.supoclip_wait_for_task("task", timeout_seconds=.05, poll_interval_seconds=.001)))
    assert time.monotonic() - started < .5
    assert cancelled == [True]
    assert result["timed_out"] is True
    assert result["task"] == (first_status or {})
    assert result["status"] == ("processing" if first_status else "unknown")


def test_stalled_progress_report_is_also_bounded(monkeypatch):
    client(monkeypatch, AsyncMock(return_value={"status": "processing", "progress": 50}))

    async def report_progress(**kwargs):
        await asyncio.Event().wait()

    started = time.monotonic()
    result = json.loads(asyncio.run(server.supoclip_wait_for_task("task", timeout_seconds=.05, ctx=SimpleNamespace(report_progress=report_progress))))
    assert time.monotonic() - started < .5
    assert result["timed_out"] is True
    assert result["progress"] == 50


@pytest.mark.parametrize("status", sorted(server.TERMINAL_STATES))
def test_terminal_states_return_without_sleeping(monkeypatch, status):
    snapshot = {"status": status, "progress": 100, "clips": []}
    client(monkeypatch, AsyncMock(return_value=snapshot))
    result = json.loads(asyncio.run(server.supoclip_wait_for_task("task", timeout_seconds=5)))
    assert result == {"status": status, "progress": 100, "timed_out": False, "task": snapshot}


def test_http_timeout_returns_last_state_but_other_api_errors_stay_errors(monkeypatch):
    client(monkeypatch, AsyncMock(side_effect=[{"status": "processing", "progress": 10}, SupoClipTimeoutError("slow")]))
    result = json.loads(asyncio.run(server.supoclip_wait_for_task("task", timeout_seconds=5, poll_interval_seconds=.001)))
    assert result["timed_out"] is True
    assert result["progress"] == 10
    client(monkeypatch, AsyncMock(side_effect=SupoClipError("Not found")))
    assert asyncio.run(server.supoclip_wait_for_task("missing")) == "Error: Not found"


def test_caller_cancellation_propagates(monkeypatch):
    client(monkeypatch, AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.supoclip_wait_for_task("task"))


@pytest.mark.parametrize("requested, expected", [(None, 30), (5, 5), (60, 30)])
def test_http_client_uses_the_smaller_timeout(monkeypatch, requested, expected):
    seen = []
    original = httpx.AsyncClient

    def build(**kwargs):
        seen.append(kwargs["timeout"])
        return original(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", build)
    asyncio.run(configured_client().request("GET", "/tasks/task", timeout_seconds=requested))
    assert seen == [expected]
