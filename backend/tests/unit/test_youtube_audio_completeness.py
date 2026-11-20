"""A completed download/checkpoint must cover the original audio timeline."""

import json
import shutil
import subprocess
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pytest

from src import youtube_proxy as proxy, youtube_source as source
from src.media.source_limits import probe_media_duration
from tests.unit.test_youtube_direct_sections import direct, VIDEO_ID, URL


@pytest.fixture
def audio(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    piece = tmp_path / "piece.aac"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "20", "-c:a", "aac", "-f", "adts", str(piece),
    ], check=True, capture_output=True)
    full = tmp_path / "full.aac"
    full.write_bytes(piece.read_bytes() * 3)
    return piece, full


def metadata():
    return {"duration": 60, "was_live": True, "formats": [
        {"format_id": "140", "protocol": "https", "url": "https://rr.example.googlevideo.com/audio",
         "ext": "m4a", "vcodec": "none", "acodec": "aac", "abr": 128},
        {"format_id": "303", "protocol": "https", "height": 1080,
         "vcodec": "avc1", "acodec": "none", "filesize": 11_460_000_000},
    ]}


def fake_transfer(monkeypatch, audio, *, always_short=False):
    piece, full = audio
    calls = []
    monkeypatch.setattr(proxy, "extract_video_info", lambda *a, **kw: metadata())

    def transfer(options, url, video_id, fmt, stem, fragments=None):
        calls.append(stem)
        path = stem.with_suffix(".m4a")
        shutil.copyfile(piece if always_short or len(calls) == 1 else full, path)
        options["_youtube_byte_budget"].add(path.stat().st_size)
        return path

    monkeypatch.setattr(proxy, "native_download_format", transfer)
    return calls


def test_truncated_audio_retries_before_success_or_cache_publication(direct, audio, monkeypatch, tmp_path, caplog):
    calls = fake_transfer(monkeypatch, audio)
    budget = proxy.ByteBudget(2_000_000)
    with caplog.at_level("INFO"):
        result = source.download_audio_for_job(URL, VIDEO_ID, budget=budget, task_id="new-task")
    assert len(calls) == 2
    assert probe_media_duration(result) == pytest.approx(60, abs=.3)
    assert budget.snapshot() == sum(path.stat().st_size for path in audio)
    successes = [record for record in caplog.records if getattr(record, "youtube_outcome", None) == "success"]
    assert len(successes) == 1
    manifest = json.loads((result.parent / "manifest.json").read_text())
    assert manifest["audio"] == result.name and manifest["full"] is None
    assert not list(tmp_path.rglob("*.part*"))


def test_persistent_truncation_never_enters_transcription_or_full_fallback(direct, audio, monkeypatch, tmp_path):
    calls = fake_transfer(monkeypatch, audio, always_short=True)
    full = Mock(side_effect=AssertionError("Full video must not start"))
    timeline = Mock(side_effect=AssertionError("Full timeline must not start"))
    monkeypatch.setattr(source, "_full_timeline_via_sections", timeline)
    with pytest.raises(proxy.AudioDownloadInterrupted, match="No full video was started"):
        source.download_analysis_source(URL, VIDEO_ID, budget=proxy.ByteBudget(2_000_000), duration=60, download_full=full)
    assert len(calls) == 2
    full.assert_not_called()
    timeline.assert_not_called()
    directory = tmp_path / "youtube-cache" / VIDEO_ID
    assert json.loads((directory / "manifest.json").read_text())["audio"] is None
    assert not list(directory.rglob("*.m4a"))


def test_old_truncated_cache_is_rejected_before_reuse(direct, audio, monkeypatch, tmp_path):
    piece, _ = audio
    directory = source.cache_dir_for(VIDEO_ID)
    directory.mkdir(parents=True)
    old = directory / f"audio-{source.AUDIO_POLICY}.m4a"
    shutil.copyfile(piece, old)
    old.with_suffix(".transcript_cache.json").write_text('{"words": [{"text": "stale"}]}')
    manifest = source._load_manifest(directory, VIDEO_ID)
    manifest["audio"] = old.name
    source._save_manifest(directory, manifest)
    calls = fake_transfer(monkeypatch, audio)
    result = source.download_audio_for_job(URL, VIDEO_ID, budget=proxy.ByteBudget(2_000_000), expected_duration=60)
    assert len(calls) == 2
    assert probe_media_duration(result) == pytest.approx(60, abs=.3)
    assert not result.with_suffix(".transcript_cache.json").exists()


@pytest.mark.parametrize("actual,accepted", [(37864, True), (37834, True), (37833, False), (3026, False), (None, False)])
def test_long_audio_requires_ffprobe_with_at_most_thirty_seconds_missing(monkeypatch, actual, accepted):
    monkeypatch.setattr(proxy, "probe_media_duration", lambda _path: actual)
    if accepted:
        assert proxy.validate_audio_duration(Path("audio.mp4"), 37864) == actual
    else:
        with pytest.raises(proxy.AudioDownloadIncomplete, match="duration mismatch"):
            proxy.validate_audio_duration(Path("audio.mp4"), 37864)


@pytest.fixture
def hls_audio(direct, audio, monkeypatch):
    piece, _ = audio
    paths = Counter()
    remaining_failures = [2]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            paths[self.path] += 1
            if self.path == "/1.aac" and remaining_failures[0]:
                remaining_failures[0] -= 1
                self.send_error(503)
                return
            payload = piece.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    info = {"duration": 60, "was_live": True, "formats": [
        {"format_id": "233", "protocol": "m3u8_native", "url": base + "/manifest",
         "ext": "mp4", "vcodec": "none", "acodec": "aac", "abr": 48,
         "fragments": [{"url": f"{base}/{i}.aac", "duration": 20} for i in range(3)]},
    ]}
    monkeypatch.setattr(proxy, "extract_video_info", lambda *a, **kw: info)
    original = proxy.base_proxy_params
    monkeypatch.setattr(proxy, "base_proxy_params", lambda *a, **kw: dict(original(*a, **kw), fixup="never", noprogress=True))
    try:
        yield paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("corrupt_index", [False, True])
def test_hls_checkpoint_from_another_task_keeps_only_verified_prefix(direct, hls_audio, tmp_path, corrupt_index):
    first = proxy.ByteBudget(2_000_000)
    with pytest.raises(proxy.AudioDownloadInterrupted):
        source.download_audio_for_job(URL, VIDEO_ID, task_id="old-task", budget=first)
    state = next(tmp_path.rglob("*.ytdl"))
    record = json.loads(state.read_text())
    assert record["downloader"]["current_fragment"]["index"] == 1
    if corrupt_index:
        # A nonempty .part with the wrong index is accepted by yt-dlp itself.
        record["downloader"]["current_fragment"]["index"] = 2
        state.write_text(json.dumps(record))
    second = proxy.ByteBudget(2_000_000)
    result = source.download_audio_for_job(URL, VIDEO_ID, task_id="other-task", budget=second)
    assert probe_media_duration(result) == pytest.approx(60, abs=.3)
    assert hls_audio["/0.aac"] == (2 if corrupt_index else 1)
    assert hls_audio["/2.aac"] == 1
    assert not list(result.parent.rglob("*.part*"))
    assert not list(result.parent.rglob("*.ytdl"))


def test_partial_timed_rendition_is_not_treated_as_full_playlist(direct, monkeypatch, tmp_path):
    info = metadata()
    info["formats"] = [{"format_id": "233", "protocol": "m3u8_native", "ext": "mp4",
                        "vcodec": "none", "acodec": "aac", "abr": 48,
                        "fragments": [{"url": "https://rr.example.googlevideo.com/first", "duration": 20}]}]
    monkeypatch.setattr(proxy, "extract_video_info", lambda *a, **kw: info)
    transfer = Mock(side_effect=AssertionError("Incomplete playlist must not transfer media"))
    monkeypatch.setattr(proxy, "native_download_format", transfer)
    with pytest.raises(proxy.AudioDownloadIncomplete, match="playlist is incomplete"):
        proxy.download_audio(URL, VIDEO_ID, tmp_path / "audio", budget=proxy.ByteBudget(2_000_000))
    transfer.assert_not_called()


@pytest.mark.asyncio
async def test_verified_source_duration_is_checkpointed_before_transcription(direct, audio, monkeypatch):
    from unittest.mock import AsyncMock
    from src.services import video_service as service

    direct.max_video_duration = 50000
    monkeypatch.setattr(service, "cached_full_media", lambda *a: None)
    monkeypatch.setattr(service, "async_get_youtube_video_info", AsyncMock(return_value={"duration": 3026}))
    monkeypatch.setattr(service, "run_download_in_thread", AsyncMock(return_value=(audio[1], True)))
    monkeypatch.setattr(service.VideoService, "_get_file_duration", lambda path: 37863.9)
    monkeypatch.setattr(service, "cleanup_downloaded_files", lambda *a, **kw: None)
    transcript = AsyncMock(side_effect=AssertionError("Duration must be saved before transcription"))
    monkeypatch.setattr(service.VideoService, "generate_transcript", transcript)
    saved = []

    class Recorded(Exception):
        pass

    async def checkpoint(patch):
        saved.append(patch)
        raise Recorded()

    with pytest.raises(Recorded):
        await service.VideoService.process_video_complete(
            url=URL, source_type="youtube", checkpoint_callback=checkpoint,
        )
    assert saved == [{"verified_source_duration_seconds": 37864}]
    transcript.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("proven", [False, True])
async def test_legacy_transcript_and_all_plan_checkpoints_require_complete_source_proof(direct, audio, monkeypatch, proven):
    from unittest.mock import AsyncMock
    from src.services import video_service as service

    path = audio[1]
    old_text = "Old paid transcript"
    old_data = {"words": [{"text": "old", "start": 0, "end": 20000}]}
    old = {"transcript": old_text, "transcript_data": old_data,
           "analysis": {"old": True}, "segments": [{"old": True}],
           "candidates": [{"old": True}], "workflow_final": {"old": True},
           "completed_positions": [1, 2]}
    if proven:
        proof = service._audio_transcript_proof(60, old_text)
        old["youtube_audio_proof"] = proof
        old_data["youtube_audio_proof"] = proof
    else:
        path.with_suffix(".transcript_cache.json").write_text(json.dumps(old_data))
    monkeypatch.setattr(service, "cached_full_media", lambda *a: None)
    monkeypatch.setattr(service, "async_get_youtube_video_info", AsyncMock(return_value={"duration": 60}))
    monkeypatch.setattr(service, "run_download_in_thread", AsyncMock(return_value=(path, True)))
    monkeypatch.setattr(service.VideoService, "_get_file_duration", lambda path: 60)
    monkeypatch.setattr(service.VideoService, "prepare_source", AsyncMock(side_effect=lambda path: path))
    monkeypatch.setattr(service, "cleanup_downloaded_files", lambda *a, **kw: None)
    monkeypatch.setattr(service, "persist_cache_sidecar", lambda *a: None)
    monkeypatch.setattr(service, "MediaStorage", lambda: type("Storage", (), {"save_source_metadata": lambda *a: None})())

    async def generate(path, **kwargs):
        assert not path.with_suffix(".transcript_cache.json").exists()
        path.with_suffix(".transcript_cache.json").write_text(json.dumps({
            "words": [{"text": "fresh", "start": 0, "end": 59000}],
        }))
        return "Fresh complete transcript"

    generate_mock = AsyncMock(side_effect=generate)
    monkeypatch.setattr(service.VideoService, "generate_transcript", generate_mock)

    class Selected(Exception):
        pass

    async def select(**kwargs):
        checkpoint = kwargs["checkpoint"]
        if proven:
            assert kwargs["transcript"] == old_text
            assert checkpoint["analysis"] == {"old": True}
            assert checkpoint["completed_positions"] == [1, 2]
        else:
            assert kwargs["transcript"] == "Fresh complete transcript"
            assert kwargs["cached_analysis_json"] is None
            assert all(checkpoint[key] is None for key in ("analysis", "segments", "candidates", "workflow_final"))
            assert checkpoint["completed_positions"] == []
        assert checkpoint["youtube_audio_proof"] == service._audio_transcript_proof(60, kwargs["transcript"])
        assert checkpoint["transcript_data"]["youtube_audio_proof"] == checkpoint["youtube_audio_proof"]
        raise Selected()

    monkeypatch.setattr(service.VideoService, "_select_highlight_segments", select)
    saved = []

    async def save(patch):
        saved.append(patch)

    with pytest.raises(Selected):
        await service.VideoService.process_video_complete(
            url=URL, source_type="youtube", checkpoint=old,
            cached_transcript=old_text, cached_analysis_json='{"old":true}', checkpoint_callback=save,
        )
    assert generate_mock.await_count == (0 if proven else 1)
    if not proven:
        assert any(patch.get("completed_positions") == [] for patch in saved)
