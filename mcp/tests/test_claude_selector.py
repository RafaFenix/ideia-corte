import asyncio
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import stat
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supoclip_mcp import claude_selector as cs
from supoclip_mcp.client import SupoClipError
from supoclip_mcp.config import load_settings

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def _task(task_id, created, *, progress=50, age=0, status=cs.AWAITING):
    return {
        "id": task_id, "status": status, "progress": progress, "created_at": created,
        "updated_at": (NOW - timedelta(seconds=age)).isoformat(),
    }


def test_pick_task_takes_the_oldest_unclaimed_and_resumes_orphans():
    tasks = [
        _task("new", "2026-11-13T11:00"),
        _task("claimed", "2026-11-13T09:00", progress=55, age=30),
        _task("old", "2026-11-13T10:00"),
        _task("done", "2026-11-13T08:00", status="completed"),
    ]
    assert cs.pick_task(tasks, NOW)["id"] == "old"
    orphan = [_task("orphan", "2026-11-13T09:00", progress=55, age=cs.STALE_SECONDS + 1)]
    assert cs.pick_task(orphan, NOW)["id"] == "orphan"
    assert cs.pick_task([tasks[1]], NOW) is None


def _settings(**overrides):
    values = {"api_key": None, "user_id": "user-1", "auth_secret": "s3cret-value", "api_url": "http://localhost:8000"}
    return replace(load_settings(), **{**values, **overrides})


def test_credentials_go_only_into_a_private_config_file():
    settings = _settings()
    with cs.mcp_config_file(settings) as path:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        server = json.loads(Path(path).read_text())["mcpServers"]["supoclip"]
        assert server["env"]["SUPOCLIP_AUTH_SECRET"] == "s3cret-value"
        assert server["env"]["SUPOCLIP_MCP_TOOLS"] == "supoclip_get_transcript,supoclip_submit_selection,supoclip_get_task"
        argv = cs.claude_command("claude", "task-1", path, "sonnet")
    assert not os.path.exists(path)
    assert not any("s3cret" in arg or "user-1" in arg for arg in argv)
    for flag in ("--strict-mcp-config", "--setting-sources", "--system-prompt", "--tools", "--disable-slash-commands"):
        assert flag in argv


def test_claude_runs_without_supoclip_credentials_or_an_api_key(monkeypatch):
    monkeypatch.setenv("SUPOCLIP_AUTH_SECRET", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = cs.claude_env()
    assert "SUPOCLIP_AUTH_SECRET" not in env and "ANTHROPIC_API_KEY" not in env and env["PATH"] == "/usr/bin"


def test_secret_falls_back_to_the_repository_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\nBACKEND_AUTH_SECRET='from-env-file'\n")
    settings = cs.resolve_settings(_settings(auth_secret=None), None, env_file)
    assert settings.auth_secret == "from-env-file" and settings.user_id == "user-1"


def test_a_second_selector_refuses_to_start(tmp_path):
    lock = tmp_path / "selector.lock"
    with cs.single_instance(lock):
        with pytest.raises(SystemExit):
            with cs.single_instance(lock):
                pass


def _selector(status_after, results):
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path, kwargs.get("json_body")))
        if method == "GET":
            return {"status": status_after.pop(0)}
        return {}

    selector = cs.Selector(
        AsyncMock(request=request), _settings(), claude="claude", model="sonnet", timeout=5,
    )
    selector.run_claude = AsyncMock(side_effect=results)
    return selector, calls


def test_a_selection_that_went_in_is_not_failed():
    selector, calls = _selector(["queued"], [({"num_turns": 4}, False)])
    assert asyncio.run(selector.handle("task-1")) == "selected"
    assert not any(path.endswith("/fail") for _, path, _ in calls)


def test_timeouts_retry_once_then_fail_with_a_clear_message():
    selector, calls = _selector([cs.AWAITING, cs.AWAITING], [({}, True), ({}, True)])
    assert asyncio.run(selector.handle("task-1")) == "failed"
    assert selector.run_claude.await_count == 2
    method, path, body = calls[-1]
    assert path == "/tasks/task-1/selection/fail" and "a tempo" in body["message"]


def test_a_logged_out_cli_fails_without_retrying():
    selector, calls = _selector([cs.AWAITING], [({"is_error": True, "result": "Invalid API key · Please run /login"}, False)])
    assert asyncio.run(selector.handle("task-1")) == "failed"
    assert selector.run_claude.await_count == 1
    assert "login" in calls[-1][2]["message"]


def test_a_refused_claim_skips_the_task():
    async def request(method, path, **kwargs):
        raise SupoClipError("409 Task is not waiting")

    selector = cs.Selector(AsyncMock(request=request), _settings(), claude="claude", model="sonnet", timeout=5)
    selector.run_claude = AsyncMock()
    assert asyncio.run(selector.handle("task-1")) == "skipped"
    selector.run_claude.assert_not_awaited()


def test_tool_allowlist_hides_every_other_tool(monkeypatch):
    from supoclip_mcp import server

    removed = []
    monkeypatch.setattr(server.mcp, "remove_tool", removed.append)
    server.limit_tools("supoclip_get_task")
    names = {tool.name for tool in server.mcp._tool_manager.list_tools()}
    assert "supoclip_get_task" not in removed
    assert set(removed) == names - {"supoclip_get_task"}
    removed.clear()
    server.limit_tools("")
    assert removed == []
