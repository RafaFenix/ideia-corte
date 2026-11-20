"""Verified audio length corrects both counters, without resurrecting charges."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from src.repositories.task_repository import TaskRepository


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["processing", "cancelled", "completed"])
async def test_verified_audio_updates_only_active_task_and_preserves_charge_state(status):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Explicit TEST_DATABASE_URL for an isolated QA database required")
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as connection:
            # Connection-local tables shadow the real schema. No application
            # rows, constraints or other connections are modified by this test.
            await connection.execute(text("""
                CREATE TEMP TABLE tasks (
                    id TEXT PRIMARY KEY, user_id TEXT, status TEXT,
                    source_duration_seconds INTEGER, updated_at TIMESTAMPTZ
                )
            """))
            await connection.execute(text("""
                CREATE TEMP TABLE minute_usage_events (
                    task_id TEXT PRIMARY KEY, user_id TEXT, minutes INTEGER,
                    source_duration_seconds INTEGER, created_at TIMESTAMPTZ,
                    refunded_at TIMESTAMPTZ
                )
            """))
            await connection.execute(text("""
                INSERT INTO tasks VALUES ('job', 'owner', :status, 3026, CURRENT_TIMESTAMP)
            """), {"status": status})
            await connection.execute(text("""
                INSERT INTO minute_usage_events VALUES (
                    'job', 'owner', 51, 3026, '2026-11-01', '2026-11-02'
                )
            """))
            async with AsyncSession(bind=connection) as db:
                await TaskRepository.record_verified_source_duration(db, "job", 37863.9)
                await TaskRepository.record_verified_source_duration(db, "job", 37863.9)
                task_seconds = await db.scalar(text("SELECT source_duration_seconds FROM tasks"))
                charge = (await db.execute(text("SELECT * FROM minute_usage_events"))).mappings().one()
            expected = 37864 if status == "processing" else 3026
            assert task_seconds == charge["source_duration_seconds"] == expected
            assert charge["minutes"] == (632 if status == "processing" else 51)
            assert charge["created_at"].day == 1 and charge["refunded_at"].day == 2
    finally:
        await engine.dispose()
