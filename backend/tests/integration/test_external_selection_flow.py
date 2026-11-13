"""Regression: a task created with selection=external must stop for the selector.

Goes through the real path (POST /tasks -> source_metadata in Postgres -> the
worker's TaskService.process_task with the real repositories); only the video
pipeline is replaced, so nothing is downloaded or sent to an LLM.
"""

import json

import pytest
from sqlalchemy import text

from src.external_selection import AWAITING_SELECTION, AwaitingSelection
from src.services.task_service import TaskService
from tests.fixtures.factories import create_user

YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def youtube_duration(monkeypatch):
    async def info(_url, **_kwargs):
        return {"duration": 600, "title": "Selector source"}

    monkeypatch.setattr("src.api.routes.tasks.async_get_youtube_video_info", info)


async def _run_worker(app, db_session, task_id):
    """What process_video_task does with the queued job's arguments."""
    job = next(job for job in reversed(app.state.queue_adapter.enqueued_jobs) if job["args"][0] == task_id)
    args = job["args"]
    service = TaskService(db_session)
    seen = {}

    async def pipeline(**kwargs):
        seen.update(kwargs)
        if kwargs.get("external_selection"):
            raise AwaitingSelection()
        raise RuntimeError("the server-side selection pipeline ran")

    service.video_service.process_video_complete = pipeline
    try:
        await service.process_task(
            task_id=args[0], url=args[1], source_type=args[2], user_id=args[3], clip_length=args[-1],
        )
    except RuntimeError:
        pass
    status = (
        await db_session.execute(text("SELECT status FROM tasks WHERE id = :id"), {"id": task_id})
    ).scalar()
    return seen, status


@pytest.mark.asyncio
@pytest.mark.parametrize(("selection", "external"), [("external", True), (None, False)])
async def test_created_task_reaches_the_worker_with_its_selection_mode(
    client, app, db_session, auth_headers, youtube_duration, selection, external
):
    await create_user(db_session, user_id="user-1")
    body = {"source": {"url": YOUTUBE_URL}, **({"selection": selection} if selection else {})}
    response = await client.post(
        "/tasks/", headers={**auth_headers, "content-type": "application/json"}, content=json.dumps(body),
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]

    seen, status = await _run_worker(app, db_session, task_id)

    assert seen.get("external_selection", False) is external
    if external:
        assert seen["cached_analysis_json"] is None
        assert status == AWAITING_SELECTION
    else:
        assert status == "error"
