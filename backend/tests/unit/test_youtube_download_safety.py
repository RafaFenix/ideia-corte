"""Regressions for large playlists, fallback limits and abandoned transfers."""

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import youtube_proxy as proxy, youtube_source as source, youtube_utils as youtube
from tests.unit.test_youtube_direct_sections import direct, native, VIDEO_ID, URL


@pytest.mark.parametrize("routing_changed", [False, True])
def test_large_playlist_audio_failure_cannot_escape_to_unbounded_full(direct, native, monkeypatch, routing_changed):
    calls, _ = native
    budget = proxy.ByteBudget(12)
    budget.add(8)
    monkeypatch.setattr(source, "download_audio_for_job", Mock(side_effect=proxy.PartialDownloadUnavailable(
        "HLS playlist exceeds its size limit",
    )))
    # Reproduce a fallback provider list selected outside the direct helper.
    monkeypatch.setattr(youtube, "_provider_order", lambda *a, **kw: ["rapidapi", "apify", "yt_dlp"])
    monkeypatch.setattr(youtube, "direct_sections_enabled", lambda *a: False)
    if routing_changed:
        direct.youtube_sections_direct = False
        direct.youtube_proxy_url = "http://proxy.example:823"
    monkeypatch.setattr(youtube, "_download_youtube_video_with_ytdlp",
                        lambda *a, **kw: pytest.fail("Legacy unbounded downloader reached"))
    with pytest.raises(proxy.DownloadBudgetExceeded, match="12 byte job limit"):
        source.download_analysis_source(
            URL, VIDEO_ID, budget=budget,
            download_full=lambda: youtube.download_youtube_video(URL, budget=budget, skip_proxy=True),
        )
    assert [call[0] for call in calls] == ["full"]
    assert "[protocol=https]" in calls[0][1]["format"]
    assert calls[0][1]["concurrent_fragment_downloads"] == 1
    assert budget.snapshot() == 16
    assert not list(Path(direct.temp_dir).glob("proxy-*"))


def _large_playlist(monkeypatch, native):
    _, ydl = native
    body = b"#EXTM3U\n" + b"# metadata padding\n" * 240_000 + b"#EXTINF:43200,\naudio.ts\n#EXT-X-ENDLIST\n"
    reads = []
    closed = []

    class Response:
        url = "https://rr1---sn-example.googlevideo.com/audio.m3u8"
        def __init__(self):
            self.body = io.BytesIO(body)
        def read(self, size):
            reads.append(size)
            return self.body.read(size)
        def close(self):
            closed.append(True)

    monkeypatch.setattr(ydl, "urlopen", lambda *a: Response())
    return body, reads, closed


def test_long_direct_playlist_larger_than_four_mb_scales_with_duration(direct, native, monkeypatch):
    body, reads, closed = _large_playlist(monkeypatch, native)
    budget = proxy.ByteBudget(10_000_000)
    result = proxy.expand_hls_format(
        {"url": "https://rr1---sn-example.googlevideo.com/audio.m3u8", "protocol": "m3u8_native"},
        {}, budget, 43200,
    )
    assert len(body) > 4_000_000
    assert result["fragments"][0]["duration"] == 43200
    assert budget.snapshot() == len(body)
    assert max(reads) <= 65_536 and closed == [True]


def test_large_audio_playlist_stops_at_small_budget_without_full_fallback(direct, native, monkeypatch):
    body, reads, closed = _large_playlist(monkeypatch, native)
    calls, ydl = native
    monkeypatch.setattr(ydl, "extract_info", lambda *a, **kw: {"duration": 43200, "formats": [
        {"format_id": "140", "protocol": "m3u8_native", "url": "https://rr1---sn-example.googlevideo.com/audio.m3u8",
         "vcodec": "none", "acodec": "aac", "ext": "m4a", "abr": 128},
    ]})
    full = Mock(side_effect=AssertionError("Full fallback must not run after budget exhaustion"))
    budget = proxy.ByteBudget(128_000)
    with pytest.raises(proxy.DownloadBudgetExceeded, match="128000 byte job limit"):
        source.download_analysis_source(URL, VIDEO_ID, budget=budget, duration=43200, download_full=full)
    assert budget.snapshot() == 128_001 < len(body)
    assert max(reads) <= 65_536 and closed == [True]
    assert calls == []
    full.assert_not_called()


@pytest.fixture
def slow_youtube(direct, monkeypatch):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import yt_dlp

    started = threading.Event()
    ended = threading.Event()
    state = {"bytes": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(64 * 1024 * 1024))
            self.end_headers()
            try:
                for _ in range(4096):
                    size = 1 if state.get("drip") else 16_384
                    self.wfile.write(b"x" * size)
                    self.wfile.flush()
                    state["bytes"] += size
                    started.set()
                    time.sleep(.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                ended.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}/media"
    original = yt_dlp.YoutubeDL

    class SyntheticYDL(original):
        def extract_info(self, url, download=True, **kwargs):
            if url != URL:
                return super().extract_info(url, download=download, **kwargs)
            audio = {"format_id": "140", "protocol": "http", "url": address,
                     "vcodec": "none", "acodec": "aac", "ext": "m4a", "abr": 128}
            info = {"id": VIDEO_ID, "title": "test", "duration": 60, "formats": [audio],
                    "url": address, "protocol": "http", "ext": "mp4", "format_id": "18",
                    "height": 720, "vcodec": "avc1", "acodec": "aac"}
            if download:
                self.process_info(info)
            return info

    direct.youtube_download_max_bytes = 100_000_000
    direct.youtube_metadata_provider = "yt_dlp"
    direct.youtube_data_api_key = ""
    monkeypatch.setattr(yt_dlp, "YoutubeDL", SyntheticYDL)
    try:
        yield started, ended, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in", [True, False])
@pytest.mark.parametrize("cancel_async_task", [False, True])
async def test_cancel_processing_stops_real_native_writes_and_joins_thread(
    direct, slow_youtube, monkeypatch, tmp_path, opt_in, cancel_async_task,
):
    import asyncio
    from src.services.video_service import VideoService
    started, ended, state = slow_youtube
    direct.youtube_sections_direct = opt_in
    cancelled = False

    async def should_cancel():
        return cancelled

    # Old root partials must also be reclaimed when no source was returned.
    stale = tmp_path / f"{VIDEO_ID}.f303.webm.part"
    stale.write_bytes(b"orphan")
    operation = asyncio.create_task(VideoService.process_video_complete(
        URL, "youtube", should_cancel=should_cancel,
    ))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(.05)
        if cancel_async_task:
            operation.cancel()
        else:
            cancelled = True
        expected = asyncio.CancelledError if cancel_async_task else RuntimeError
        with pytest.raises(expected):
            await asyncio.wait_for(operation, timeout=5)
        assert await asyncio.to_thread(ended.wait, 2)
        after = state["bytes"]
        await asyncio.sleep(.1)
        assert state["bytes"] == after < 2_000_000
        assert not list(tmp_path.rglob("*.part*"))
        assert not list(tmp_path.rglob("*.ytdl*"))
        assert not list(tmp_path.glob("ytdl-*"))
        assert not list(tmp_path.rglob("audio-original-v2.*"))
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in", [True, False])
async def test_real_full_fallback_transfer_stops_on_the_shared_budget(direct, slow_youtube, opt_in, monkeypatch, tmp_path):
    import asyncio
    from src.download_progress import run_download_in_thread
    direct.youtube_sections_direct = opt_in
    _, ended, state = slow_youtube
    budget = proxy.ByteBudget(128_000)
    budget.add(4_000)  # Bytes already spent before audio recovery.
    monkeypatch.setattr(source, "download_audio_for_job", Mock(side_effect=proxy.PartialDownloadUnavailable(
        "HLS playlist exceeds its size limit",
    )))
    if opt_in:
        monkeypatch.setattr(youtube, "_download_youtube_video_with_ytdlp",
                            lambda *a, **kw: pytest.fail("Unbounded legacy fallback reached"))
    with pytest.raises(proxy.DownloadBudgetExceeded, match="128000 byte job limit"):
        await run_download_in_thread(
            source.download_analysis_source, URL, VIDEO_ID, budget=budget,
            download_full=lambda: youtube.download_youtube_video(URL, budget=budget, skip_proxy=True),
        )
    assert 128_000 < budget.snapshot() <= 128_000 + 65_536
    assert await asyncio.to_thread(ended.wait, 2)
    assert state["bytes"] < 1_000_000
    assert not list(tmp_path.rglob("*.part*"))
    assert not list(tmp_path.glob("proxy-*"))
    assert not list(tmp_path.glob("ytdl-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in", [True, False])
async def test_cancellation_interrupts_a_slow_trickle_inside_a_buffered_read(direct, slow_youtube, opt_in):
    import asyncio
    import time
    from src.services.video_service import VideoService
    started, ended, state = slow_youtube
    state["drip"] = True
    direct.youtube_sections_direct = opt_in
    cancelled = False
    async def should_cancel():
        return cancelled
    operation = asyncio.create_task(VideoService.process_video_complete(URL, "youtube", should_cancel=should_cancel))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        before = time.monotonic()
        cancelled = True
        with pytest.raises(RuntimeError, match="Task cancelled"):
            await asyncio.wait_for(operation, timeout=3)
        assert time.monotonic() - before < 2
        assert await asyncio.to_thread(ended.wait, 2)
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("local_mux", [False, True])
async def test_cancelling_ytdlp_subprocess_reaps_its_group_and_keeps_other_job(tmp_path, local_mux):
    import asyncio
    import sys
    from src.download_cancellation import managed_ytdlp_client, run_local_process
    from src.download_progress import run_download_in_thread
    from yt_dlp.downloader import external

    processes = []
    cancelled = [False, False]
    paths = [tmp_path / "first.part", tmp_path / "second.part"]
    script = "import pathlib,sys,time; f=pathlib.Path(sys.argv[1]).open('wb');\nwhile True: f.write(b'x'*100); f.flush(); time.sleep(.02)"

    def write(index):
        try:
            if local_mux:
                run_local_process([sys.executable, "-c", script, str(paths[index])], capture_output=True, text=True)
                return
            with managed_ytdlp_client({"quiet": True}):
                with external.Popen([sys.executable, "-c", script, str(paths[index])]) as process:
                    processes.append(process)
                    process.wait()
        finally:
            paths[index].unlink(missing_ok=True)

    async def should_cancel(index):
        return cancelled[index]

    operations = [asyncio.create_task(run_download_in_thread(
        write, i, should_cancel=lambda i=i: should_cancel(i),
    )) for i in range(2)]
    try:
        for _ in range(200):
            if all(path.exists() and path.stat().st_size for path in paths):
                break
            await asyncio.sleep(.02)
        assert all(path.exists() for path in paths)
        cancelled[0] = True
        with pytest.raises(RuntimeError, match="Task cancelled"):
            await asyncio.wait_for(operations[0], timeout=3)
        assert not paths[0].exists()
        before = paths[1].stat().st_size
        await asyncio.sleep(.1)
        assert paths[1].stat().st_size > before and not operations[1].done()
        cancelled[1] = True
        with pytest.raises(RuntimeError, match="Task cancelled"):
            await asyncio.wait_for(operations[1], timeout=3)
        assert all(process.poll() is not None for process in processes)
        assert not any(path.exists() for path in paths)
    finally:
        cancelled[:] = [True, True]
        await asyncio.gather(*operations, return_exceptions=True)


def test_no_source_returned_still_cleans_legacy_partials_only_for_this_video(direct, tmp_path):
    from src.services.task_service import TaskService
    names = [f"{VIDEO_ID}.f303.webm.part", f"{VIDEO_ID}.f303.webm.ytdl",
             f"{VIDEO_ID}.f303.webm.part-Frag32", f"{VIDEO_ID}.part", f"{VIDEO_ID}.ytdl.tmp"]
    kept = [f"{VIDEO_ID}.transcript_cache.json", "another-vid.f303.webm.part"]
    for name in names + kept:
        (tmp_path / name).write_text("fixture")
    TaskService._cleanup_source_video(None, "youtube", URL)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(kept)


def test_orphan_sweep_recognizes_old_component_partials_and_preserves_recent(direct, tmp_path):
    import os
    import time
    from types import SimpleNamespace
    from src.workers.media_cleanup import local_orphan_candidates
    old = [tmp_path / f"{VIDEO_ID}.f303.webm.part", tmp_path / f"{VIDEO_ID}.f303.webm.ytdl"]
    recent = tmp_path / f"{VIDEO_ID}.f304.webm.part"
    transcript = tmp_path / f"{VIDEO_ID}.transcript_cache.json"
    for path in old + [recent, transcript]:
        path.write_bytes(b"fixture")
    for path in old + [transcript]:
        os.utime(path, (time.time() - 1000,) * 2)
    storage = SimpleNamespace(config=SimpleNamespace(temp_dir=str(tmp_path), media_orphan_grace_seconds=60))
    candidates = list(local_orphan_candidates(storage))
    assert sorted(candidates) == sorted((str(path), False) for path in old)


def test_explicit_proxy_keeps_precedence_over_direct_safety_flag(direct):
    direct.youtube_proxy_url = "http://proxy.example:823"
    assert proxy.ByteBudget(1000).native_only is False
    assert youtube._provider_order(direct, skip_proxy=False) == ["yt_dlp_proxy", "rapidapi", "apify", "yt_dlp"]


def test_local_probe_preserves_check_semantics_in_cancel_scope():
    import sys
    import subprocess
    from src.download_cancellation import DownloadCancellation, cancellation_scope, run_local_process
    with cancellation_scope(DownloadCancellation()):
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_local_process([sys.executable, "-c", "raise SystemExit(3)"], capture_output=True, text=True, check=True)
    assert error.value.returncode == 3


def test_cancelling_local_section_assembly_removes_owned_work_but_keeps_cache(direct, native, monkeypatch, tmp_path):
    from src.download_cancellation import DownloadCancelled
    def interrupted(slices, dest):
        dest.write_bytes(b"unfinished mux")
        raise DownloadCancelled("Task cancelled")
    monkeypatch.setattr(source, "assemble_timeline", interrupted)
    with pytest.raises(DownloadCancelled):
        source.materialize_clip_source(URL, VIDEO_ID, [{"start_time": 20, "end_time": 30}], duration=60)
    assert not list((tmp_path / "youtube-work").iterdir())
    # Completed reusable sections are independent of the interrupted timeline.
    manifest = json.loads((tmp_path / "youtube-cache" / VIDEO_ID / "manifest.json").read_text())
    assert len(manifest["ranges"]) == 1


def test_known_oversized_full_fallback_is_rejected_before_media(direct, native, monkeypatch, tmp_path):
    _, ydl = native
    budget = proxy.ByteBudget(1000)
    budget.add(100)
    def download(self, urls):
        self.options["match_filter"]({"requested_formats": [
            {"filesize": 11_460_000_000}, {"filesize": None},
        ]})
        pytest.fail("Known oversized source must not start transferring media")
    monkeypatch.setattr(ydl, "download", download)
    with pytest.raises(proxy.DownloadBudgetExceeded, match="remaining 900 byte job limit"):
        youtube.download_youtube_video(URL, budget=budget)
    assert budget.snapshot() == 100
    assert not list(tmp_path.glob("proxy-*"))


@pytest.mark.asyncio
async def test_native_full_reader_enforces_budget_even_without_progress_hooks(direct, slow_youtube, monkeypatch, tmp_path):
    import asyncio
    import yt_dlp
    from yt_dlp.networking import Request
    from src.download_progress import run_download_in_thread
    _, ended, _ = slow_youtube
    budget = proxy.ByteBudget(128_000)
    def download(self, urls):
        info = self.extract_info(URL, download=False)
        template = self.params["outtmpl"]
        if isinstance(template, dict):
            template = template["default"]
        path = Path(template.replace("%(ext)s", "mp4") + ".part")
        with self.urlopen(Request(info["url"])) as response, path.open("wb") as output:
            while chunk := response.read(16_384):
                output.write(chunk)
        pytest.fail("Reader must stop the stream without needing progress hooks")
    monkeypatch.setattr(yt_dlp.YoutubeDL, "download", download)
    with pytest.raises(proxy.DownloadBudgetExceeded):
        await run_download_in_thread(youtube.download_youtube_video, URL, budget=budget)
    assert budget.snapshot() == budget.limit + 1
    assert await asyncio.to_thread(ended.wait, 2)
    assert not list(tmp_path.rglob("*.part*"))
    assert not list(tmp_path.glob("proxy-*"))


def test_metadata_budget_failure_does_not_try_another_provider(direct, monkeypatch):
    budget = proxy.ByteBudget(100)
    def metadata(*args, **kwargs):
        budget.add(101)
    api = Mock(side_effect=AssertionError("Budget exhaustion must not recover with another metadata provider"))
    direct.youtube_metadata_provider = "yt_dlp"
    monkeypatch.setattr(youtube, "_fetch_video_info_with_ytdlp", metadata)
    monkeypatch.setattr(youtube, "_fetch_video_info_with_youtube_data_api", api)
    with pytest.raises(proxy.DownloadBudgetExceeded):
        youtube.get_youtube_video_info(URL, budget=budget)
    api.assert_not_called()
