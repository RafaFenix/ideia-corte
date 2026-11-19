"""Own fixtures for bounded upload probes; no upstream implementation is used."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
import pytest

from src.media import upload_probe as probe
from src.api.routes import tasks
from src.services.video_service import VideoService


def payload(duration="1", **changes):
    return {"format": {"format_name": "mov,mp4", "duration": duration, "start_time": "0"},
            "streams": [{"index": 0, "codec_type": "video"}], **changes}


def test_final_fragment_overrides_short_header_and_ignores_data_tracks(monkeypatch):
    calls = []
    results = [payload(), payload(packets=[
        {"stream_index": 0, "pts_time": "11.96", "duration_time": ".04"},
        {"stream_index": 9, "pts_time": "9999", "duration_time": "10"},
    ])]

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(results.pop(0)))

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.upload_duration("https://storage.test/signed") == pytest.approx(12)
    assert "-read_intervals" not in calls[0][0]
    assert "-read_intervals" in calls[1][0]
    assert 0 < calls[1][1]["timeout"] <= calls[0][1]["timeout"] <= 15


def test_both_passes_share_one_deadline(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    timeouts = []
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock.now += 10
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload()))

    monkeypatch.setattr(probe.subprocess, "run", run)
    with pytest.raises(TimeoutError):
        probe.upload_duration("upload.mp4")
    assert timeouts == [15, 5]


def test_seek_failure_never_falls_back_to_a_short_header(monkeypatch):
    responses = iter([SimpleNamespace(returncode=0, stdout=json.dumps(payload())), SimpleNamespace(returncode=1, stdout="")])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="Unable to read"):
        probe.upload_duration("upload.mp4")


@pytest.mark.parametrize("result", ["broken JSON", "[]", json.dumps(payload(packets=[]))])
def test_malformed_or_unverifiable_mp4_is_rejected(monkeypatch, result):
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=result))
    with pytest.raises(ValueError):
        probe.upload_duration("broken.mp4")


@pytest.mark.parametrize("duration", ["0", "-2", "N/A", "nan", "inf"])
def test_invalid_duration_is_rejected(monkeypatch, duration):
    metadata = {"format": {"format_name": "matroska,webm", "duration": duration}}
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(metadata)))
    with pytest.raises(ValueError):
        probe.upload_duration("broken.webm")


def test_timeout_kills_the_child_and_redacts_the_signed_url(tmp_path, monkeypatch):
    binary = tmp_path / "ffprobe"
    binary.write_text('#!/bin/sh\nexec /bin/sleep 10\n')
    binary.chmod(0o755)
    processes = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(probe, "UPLOAD_PROBE_TIMEOUT_SECONDS", .3)
    start = time.monotonic()
    with pytest.raises(TimeoutError) as error:
        probe.upload_duration("https://storage.test/video?signature=secret")
    assert time.monotonic() - start < 2
    assert "secret" not in str(error.value)
    assert len(processes) == 1
    assert processes[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)


@pytest.mark.asyncio
async def test_slow_probe_has_a_friendly_preflight_error(monkeypatch):
    async def stalled(*a, **k):
        import asyncio
        await asyncio.Event().wait()

    monkeypatch.setattr(tasks, "run_in_thread", stalled)
    monkeypatch.setattr(tasks, "UPLOAD_PREFLIGHT_TIMEOUT_SECONDS", .02)
    with pytest.raises(HTTPException) as error:
        await tasks._preflight_upload_source("upload://demo.mp4", "owner", 3600)
    assert error.value.status_code == 503
    assert error.value.detail["code"] == "VIDEO_METADATA_UNAVAILABLE"
    assert "took too long" in error.value.detail["message"]
    assert "No minutes were used" in error.value.detail["message"]


@pytest.mark.asyncio
async def test_process_timeout_reaches_the_friendly_error(monkeypatch):
    def slow(*args):
        raise TimeoutError("private details")

    monkeypatch.setattr(VideoService, "probe_upload_duration", slow)
    with pytest.raises(HTTPException) as error:
        await tasks._preflight_upload_source("upload://demo.mp4", "owner", 3600)
    assert error.value.status_code == 503
    assert "private" not in str(error.value.detail)


@pytest.mark.asyncio
async def test_timeout_creates_no_task_or_job(monkeypatch):
    monkeypatch.setattr(tasks, "_get_user_id_from_headers", AsyncMock(return_value="owner"))
    billing = AsyncMock(return_value={"max_upload_duration_seconds": 3600})
    monkeypatch.setattr(tasks.BillingService, "assert_can_create_task", billing)
    service = SimpleNamespace(video_service=VideoService(), create_task_with_source=AsyncMock())
    monkeypatch.setattr(tasks, "TaskService", lambda _: service)

    def slow(*args):
        raise TimeoutError("slow probe")

    monkeypatch.setattr(VideoService, "probe_upload_duration", slow)
    queue = SimpleNamespace(enqueue_processing_job=AsyncMock())
    request = SimpleNamespace(json=AsyncMock(return_value={"source": {"url": "upload://demo.mp4"}}),
                              app=SimpleNamespace(state=SimpleNamespace(queue_adapter=queue)))
    with pytest.raises(HTTPException) as error:
        await tasks.create_task(request, db=None)
    assert error.value.status_code == 503
    service.create_task_with_source.assert_not_awaited()
    queue.enqueue_processing_job.assert_not_awaited()
    assert all(len(call.args) == 1 for call in billing.await_args_list)


@pytest.fixture
def media(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")

    def make(*, fragmented, audio=True):
        target = tmp_path / f"{fragmented}-{audio}.mp4"
        command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=12"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=12"]
        command += ["-c:v", "libx264", "-preset", "ultrafast", "-g", "25", "-c:a", "aac"]
        if fragmented:
            command += ["-movflags", "empty_moov+frag_keyframe+default_base_moof"]
        subprocess.run([*command, str(target)], capture_output=True, check=True)
        return target

    return make


@pytest.mark.parametrize("fragmented,audio", [(True, True), (True, False), (False, True)])
def test_real_mp4_length_includes_last_fragment(media, fragmented, audio):
    path = media(fragmented=fragmented, audio=audio)
    assert probe.upload_duration(path) == pytest.approx(12, abs=.06)


@contextmanager
def storage_server(path):
    ranges = []
    data = path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range", "")
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
            start = int(match[1]) if match else 0
            end = int(match[2]) if match and match[2] else len(data) - 1
            end = min(end, len(data) - 1)
            ranges.append((start, end))
            self.send_response(206 if match else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.end_headers()
            try:
                self.wfile.write(data[start:end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/source.mp4", ranges
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fragmented_object_storage_uses_range_seeks(media):
    path = media(fragmented=True)
    with storage_server(path) as (url, ranges):
        assert probe.upload_duration(url) == pytest.approx(12, abs=.06)
    assert any(start > 0 for start, _ in ranges)
