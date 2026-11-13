import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supoclip_mcp import server

URL = "https://youtu.be/abcdefghijk"


def _client(monkeypatch, response):
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(server, "_client", lambda: SimpleNamespace(request=request))
    return request


def test_external_selection_is_sent_only_when_asked(monkeypatch):
    request = _client(monkeypatch, {"task_id": "task-1"})
    asyncio.run(server.supoclip_create_clip_task(url=URL))
    assert "selection" not in request.await_args.kwargs["json_body"]
    asyncio.run(server.supoclip_create_clip_task(url=URL, selection="external"))
    assert request.await_args.kwargs["json_body"]["selection"] == "external"


def test_external_selection_refuses_other_workflows(monkeypatch):
    request = _client(monkeypatch, {})
    result = asyncio.run(server.supoclip_create_clip_task(url=URL, workflow="recap", selection="external"))
    assert "workflow='highlights' only" in result
    request.assert_not_awaited()


def test_transcript_and_selection_tools_hit_their_routes(monkeypatch):
    request = _client(monkeypatch, {"lines": []})
    asyncio.run(server.supoclip_get_transcript("task-1", offset=200, limit=100, include_words=True))
    assert request.await_args.args == ("GET", "/tasks/task-1/transcript")
    assert request.await_args.kwargs["params"] == {"offset": 200, "limit": 100, "words": "true"}
    segments = [{"start": "00:10", "end": "00:40", "hook_title": "A strong hook"}]
    asyncio.run(server.supoclip_submit_selection("task-1", segments))
    assert request.await_args.args == ("POST", "/tasks/task-1/selection")
    assert request.await_args.kwargs["json_body"] == {"segments": segments}


def test_wait_returns_when_the_task_awaits_a_selection(monkeypatch):
    _client(monkeypatch, {"status": "awaiting_selection", "progress": 50})
    result = asyncio.run(server.supoclip_wait_for_task("task-1", timeout_seconds=5, poll_interval_seconds=2))
    assert '"timed_out": false' in result
