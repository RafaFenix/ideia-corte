"""External selection: an MCP client picks the clips instead of the backend LLM."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes import tasks
from src.external_selection import (
    AWAITING_SELECTION,
    AwaitingSelection,
    transcript_page,
    validate_selection,
)

from tests.unit.test_task_service import build_task_service

# 10-second lines over 3 minutes, five words each.
TRANSCRIPT = "\n".join(
    f"[{i // 6:02d}:{(i % 6) * 10:02d} - {(i + 1) // 6:02d}:{((i + 1) % 6) * 10:02d}] line {i} has five words"
    for i in range(18)
)


def test_transcript_page_paginates_and_limits_words_to_the_page():
    words = [{"text": f"w{s}", "start": s * 1000, "end": s * 1000 + 500} for s in range(0, 180, 5)]
    page = transcript_page(TRANSCRIPT, {"words": words}, offset=2, limit=3, include_words=True)
    assert [line["i"] for line in page["lines"]] == [2, 3, 4]
    assert page["lines"][0]["start"] == "00:20"
    assert page["total_lines"] == 18 and page["next_offset"] == 5
    # Labels are whole seconds, so the last second of the page is included.
    assert [word["t"] for word in page["words"]] == ["w20", "w25", "w30", "w35", "w40", "w45", "w50"]
    last = transcript_page(TRANSCRIPT, None, offset=15, limit=10)
    assert last["next_offset"] is None and "words" not in last


def test_selection_snaps_keeps_order_and_rejects_unusable_segments():
    segments, accepted, rejected = validate_selection(
        [
            {"start": "01:02", "end": 118, "hook_title": "Second best moment here", "scores": {"hook": 30}},
            {"start": "00:00", "end": "00:30", "hook_title": "First clip title"},
            {"start": "00:10", "end": "00:20"},  # too short: repaired, then overlaps the second pick
            {"start": "01:10", "end": "01:50"},  # overlaps the first pick
            {"start": "02:00", "end": "01:00"},
        ],
        TRANSCRIPT, "auto", max_clips=10,
    )
    assert [(s["start_time"], s["end_time"]) for s in segments] == [("01:00", "02:00"), ("00:00", "00:30")]
    assert accepted[0]["snapped"] is True and accepted[1]["snapped"] is False
    assert segments[0]["virality"]["hook_score"] == 25  # clamped to the model's 0-25 range
    assert segments[0]["virality"]["total_score"] == 25
    assert all(s["opening_reviewed"] for s in segments)
    assert "line 6 has five words" in segments[0]["text"]
    assert [entry["index"] for entry in rejected] == [2, 3, 4]
    assert "overlaps" in rejected[1]["reason"]


def test_selection_respects_the_clip_budget_and_opening_review_opt_out():
    segments, _accepted, rejected = validate_selection(
        [
            {"start": "00:00", "end": "00:30", "respect_bounds": False},
            {"start": "00:40", "end": "01:10"},
            {"start": "01:20", "end": "01:50"},
        ],
        TRANSCRIPT, "auto", max_clips=2,
    )
    # Three minutes of source still allows the minimum slate, capped by max_clips.
    assert len(segments) == 2
    assert "opening_reviewed" not in segments[0]
    assert rejected == [{"index": 2, "reason": "over max_clips (2)"}]


def _external_service():
    service = build_task_service()
    # Like the real row: get_task_by_id never carries source_metadata.
    service.task_repo.get_task_by_id = AsyncMock(return_value={"status": "queued", "include_broll": True})
    service.task_repo.load_source_metadata = AsyncMock(return_value={"selection_mode": "external"})
    return service


@pytest.mark.asyncio
async def test_external_task_stops_in_awaiting_selection_without_the_shared_analysis():
    service = _external_service()
    service.cache_repo.get_cache = AsyncMock(return_value={"transcript_text": "T", "analysis_json": '{"x": 1}'})
    service.video_service.process_video_complete = AsyncMock(side_effect=AwaitingSelection())
    result = await service.process_task(task_id="task-1", url="upload://a.mp4", source_type="upload")
    kwargs = service.video_service.process_video_complete.await_args.kwargs
    assert kwargs["external_selection"] is True
    assert kwargs["cached_analysis_json"] is None
    assert result["status"] == AWAITING_SELECTION
    assert service.task_repo.update_task_status.await_args_list[-1].args[2] == AWAITING_SELECTION
    service.cache_repo.upsert_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_render_never_writes_the_shared_analysis_or_plans_broll():
    service = _external_service()
    await service.process_task(task_id="task-1", url="upload://a.mp4", source_type="upload")
    (write,) = service.cache_repo.upsert_cache.await_args_list
    assert write.kwargs["analysis_json"] is None
    assert service.video_service.create_single_clip.await_args.kwargs["include_broll"] is False


@pytest.mark.asyncio
async def test_default_task_keeps_its_pipeline_arguments():
    service = build_task_service()
    await service.process_task(task_id="task-1", url="upload://a.mp4", source_type="upload")
    assert "external_selection" not in service.video_service.process_video_complete.await_args.kwargs


@pytest.fixture
def selection_routes(monkeypatch):
    task = {
        "id": "task-1", "user_id": "user-1", "status": AWAITING_SELECTION,
        "source_url": "upload://a.mp4", "source_type": "upload", "processing_mode": "fast",
    }
    repo = SimpleNamespace(
        get_task_by_id=AsyncMock(return_value=task),
        update_task_status=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(tasks, "TaskService", lambda _db: SimpleNamespace(task_repo=repo))
    monkeypatch.setattr(tasks, "_require_task_owner", AsyncMock(return_value=task))
    monkeypatch.setattr(tasks, "_save_task_source_metadata", AsyncMock())
    monkeypatch.setattr(
        tasks, "_load_task_source_metadata",
        AsyncMock(return_value={"selection_mode": "external", "clip_length": "auto"}),
    )

    @asynccontextmanager
    async def guard(_db, _task_id):
        yield

    monkeypatch.setattr(tasks, "task_run_guard", guard)
    monkeypatch.setattr(tasks.BillingService, "assert_can_resume_task", AsyncMock())
    queue = AsyncMock(return_value="job-1")
    monkeypatch.setattr(tasks.JobQueue, "enqueue_processing_job", queue)
    monkeypatch.setattr(tasks.redis, "Redis", lambda **_kw: SimpleNamespace(delete=AsyncMock(), aclose=AsyncMock()))
    checkpoints = SimpleNamespace(
        get=AsyncMock(return_value={"transcript": TRANSCRIPT, "transcript_data": {}}),
        save=AsyncMock(),
    )
    monkeypatch.setattr(tasks, "TaskCheckpointRepository", checkpoints)
    return task, repo, queue, checkpoints


def _body(monkeypatch, body):
    monkeypatch.setattr(tasks, "_read_json_object", AsyncMock(return_value=body))


@pytest.mark.asyncio
async def test_submitted_selection_becomes_the_analysis_and_requeues(selection_routes, monkeypatch):
    _task, repo, queue, checkpoints = selection_routes
    _body(monkeypatch, {"segments": [{"start": "00:00", "end": "00:30", "hook_title": "A good title"}]})
    result = await tasks.submit_task_selection("task-1", SimpleNamespace(), db=None)
    assert result["status"] == "queued" and result["job_id"] == "job-1"
    patch = checkpoints.save.await_args.args[3]
    assert patch["analysis"]["most_relevant_segments"][0]["hook_title"] == "A good title"
    assert patch["segments"] is None and patch["candidates"] is None
    assert repo.update_task_status.await_args_list[0].args[2] == "queued"
    queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_selection_is_refused_outside_awaiting_selection(selection_routes, monkeypatch):
    task, _repo, queue, checkpoints = selection_routes
    task["status"] = "processing"
    _body(monkeypatch, {"segments": [{"start": "00:00", "end": "00:30"}]})
    with pytest.raises(HTTPException) as error:
        await tasks.submit_task_selection("task-1", SimpleNamespace(), db=None)
    assert error.value.status_code == 409
    checkpoints.save.assert_not_awaited()
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_rejected_selection_returns_reasons_and_queues_nothing(selection_routes, monkeypatch):
    _task, _repo, queue, checkpoints = selection_routes
    _body(monkeypatch, {"segments": [{"start": "10:00", "end": "10:30"}]})  # past the transcript
    with pytest.raises(HTTPException) as error:
        await tasks.submit_task_selection("task-1", SimpleNamespace(), db=None)
    assert error.value.status_code == 422
    assert error.value.detail["rejected"][0]["index"] == 0
    checkpoints.save.assert_not_awaited()
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcript_route_returns_a_page_with_limits(selection_routes):
    page = await tasks.get_task_transcript("task-1", SimpleNamespace(), offset=0, limit=5, words=False, db=None)
    assert len(page["lines"]) == 5 and page["next_offset"] == 5
    assert page["limits"]["min_seconds"] == 15 and page["limits"]["max_clips"] >= 1


@pytest.mark.asyncio
async def test_resume_still_refuses_a_task_awaiting_selection(selection_routes):
    with pytest.raises(HTTPException) as error:
        await tasks.resume_task("task-1", SimpleNamespace(), db=None)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_claim_marks_the_task_and_refuses_other_states(selection_routes):
    _task, repo, _queue, _checkpoints = selection_routes
    assert (await tasks.claim_task_selection("task-1", SimpleNamespace(), db=None))["claimed"] is True
    call = repo.update_task_status.await_args
    assert call.args[2] == AWAITING_SELECTION and call.kwargs["progress"] == tasks.SELECTION_CLAIMED_PROGRESS
    assert call.kwargs["expected_statuses"] == [AWAITING_SELECTION]
    repo.update_task_status.return_value = False
    with pytest.raises(HTTPException) as error:
        await tasks.claim_task_selection("task-1", SimpleNamespace(), db=None)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_fail_records_a_clear_message_and_error_code(selection_routes, monkeypatch):
    _task, repo, _queue, _checkpoints = selection_routes
    repo.update_task_runtime_metadata = AsyncMock()
    _body(monkeypatch, {"message": "Claude CLI sem login"})
    result = await tasks.fail_task_selection("task-1", SimpleNamespace(), db=None)
    assert result["error_code"] == "selection_failed"
    call = repo.update_task_status.await_args
    assert call.args[2] == "error" and call.kwargs["progress_message"] == "Claude CLI sem login"
    assert repo.update_task_runtime_metadata.await_args.kwargs["error_code"] == "selection_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("self_host,admin,expected", [(True, False, True), (False, True, True), (False, False, False)])
async def test_workflows_info_says_whether_external_selection_is_offered(monkeypatch, self_host, admin, expected):
    monkeypatch.setattr(tasks, "_get_user_id_from_headers", AsyncMock(return_value="user-1"))
    monkeypatch.setattr(tasks, "allowed_workflows", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(tasks, "get_config", lambda: SimpleNamespace(self_host=self_host))
    monkeypatch.setattr(tasks, "user_is_admin", AsyncMock(return_value=admin))
    info = await tasks.get_workflows(SimpleNamespace(headers={}), db=object())
    assert info["external_selection"] is expected
    assert info["workflows"][0]["id"] == "highlights"
