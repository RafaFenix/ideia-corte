"""Direct audio uses an idle clock, resumes partials and avoids oversized recovery."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from yt_dlp.networking import Request
from yt_dlp.utils import DownloadError

from src import youtube_proxy as proxy, youtube_source as source
from tests.unit.test_youtube_direct_sections import direct, native, VIDEO_ID, URL


def test_slow_hls_audio_keeps_progress_beyond_initial_total_deadline(direct, native, monkeypatch, tmp_path):
    _, ydl = native
    now = [0.0]
    attempts = []
    metadata = []
    monkeypatch.setattr(proxy.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(proxy, "probe_media_duration", lambda _path: 37864.0)
    formats = [
        {"format_id": "140", "protocol": "https", "url": "https://rr.example.googlevideo.com/a",
         "ext": "m4a", "vcodec": "none", "acodec": "aac", "abr": 128},
        {"format_id": "233", "protocol": "m3u8_native", "url": "https://rr.example.googlevideo.com/a.m3u8",
         "ext": "mp4", "vcodec": "none", "acodec": "aac", "abr": 48,
         "fragments": [{"url": "https://rr.example.googlevideo.com/seg", "duration": 37864}]},
    ]
    def info(*a, **kw):
        metadata.append(1)
        return {"duration": 37864, "formats": formats}
    monkeypatch.setattr(ydl, "extract_info", info)
    monkeypatch.setattr(proxy, "contain_raw_aac", lambda path: path)

    class Response:
        def __init__(self):
            self.count = 0
        def read(self, size):
            if self.count == 6:
                return b""
            self.count += 1
            now[0] += 100  # each read progresses; total audio transfer is 600s
            return b"x"
        def close(self):
            pass

    monkeypatch.setattr(ydl, "urlopen", lambda *a: Response())
    def process(self, info):
        attempts.append(info["format_id"])
        if info["format_id"] == "140":
            raise DownloadError("HTTP Error 403: Forbidden")
        target = Path(self.options["outtmpl"].replace("%(ext)s", info["ext"]))
        # No progress hooks: raw response bytes must advance the idle clock.
        with self.urlopen(Request(info["url"])) as response, target.open("wb") as output:
            while chunk := response.read(65_536):
                output.write(chunk)
    monkeypatch.setattr(ydl, "process_info", process)
    budget = proxy.ByteBudget(1000)
    result = proxy.download_audio(URL, VIDEO_ID, tmp_path / "audio", budget=budget)
    assert result.read_bytes() == b"xxxxxx"
    assert now[0] == 600 > direct.youtube_proxy_timeout_seconds + 60
    assert attempts == ["140", "233"] and len(metadata) == 1
    assert budget.snapshot() == 6


def test_duplicate_progress_does_not_extend_audio_idle_deadline(direct, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(proxy.time, "monotonic", lambda: now[0])
    options = proxy.download_options(None, VIDEO_ID, proxy.ByteBudget(1000))
    options["_youtube_audio_clock"] = proxy._AudioTransferClock(300, 300)
    hook = options["progress_hooks"][0]
    hook({"status": "downloading", "filename": "audio", "downloaded_bytes": 1})
    now[0] = 299
    hook({"status": "downloading", "filename": "audio", "downloaded_bytes": 1})
    now[0] = 301
    with pytest.raises(proxy.DownloadDeadlineExceeded, match="without new data"):
        hook({"status": "downloading", "filename": "audio", "downloaded_bytes": 1})


def test_hls_playlist_socket_timeout_remains_classified_as_audio_timeout(direct, native, monkeypatch):
    _, ydl = native
    monkeypatch.setattr(ydl, "urlopen", Mock(side_effect=TimeoutError("Read timed out")))
    options = proxy.download_options(None, VIDEO_ID, proxy.ByteBudget(1000))
    options["_youtube_audio_clock"] = proxy._AudioTransferClock(300, proxy.time.monotonic() + 300)
    with pytest.raises(proxy.DownloadDeadlineExceeded, match="audio playlist transfer timed out"):
        proxy.expand_hls_format({"url": "https://rr.example.googlevideo.com/a.m3u8", "protocol": "m3u8_native"}, options, proxy.ByteBudget(1000), 37864)


@pytest.mark.parametrize("estimate,resumable", [(11_460_000_000, False), (None, False), (1000, True)])
def test_direct_audio_interruption_refuses_unsafe_or_wasteful_full_timeline(direct, monkeypatch, estimate, resumable):
    budget = proxy.ByteBudget(2_000_000_000)
    budget.add(48_974_170)
    failure = proxy.AudioDownloadInterrupted(
        "Read timed out", full_estimate_bytes=estimate, resumable=resumable,
    )
    monkeypatch.setattr(source, "download_audio_for_job", Mock(side_effect=failure))
    timeline = Mock(side_effect=AssertionError("Full timeline must not start"))
    full = Mock(side_effect=AssertionError("Full provider chain must not start"))
    monkeypatch.setattr(source, "_full_timeline_via_sections", timeline)
    with pytest.raises(proxy.AudioDownloadInterrupted, match="No full video was started"):
        source.download_analysis_source(URL, VIDEO_ID, duration=37864, budget=budget, download_full=full)
    assert budget.snapshot() == 48_974_170
    timeline.assert_not_called()
    full.assert_not_called()


def test_wrapped_hls_timeout_without_estimate_does_not_download_full(direct, monkeypatch):
    failure = proxy.AudioDownloadForbidden("HLS audio download failed")
    failure.__cause__ = TimeoutError("Read timed out (read timeout=0.13s)")
    monkeypatch.setattr(source, "download_audio_for_job", Mock(side_effect=failure))
    timeline = Mock(side_effect=AssertionError("No full timeline after unknown-size timeout"))
    full = Mock(side_effect=AssertionError("No full file after unknown-size timeout"))
    monkeypatch.setattr(source, "_full_timeline_via_sections", timeline)
    with pytest.raises(proxy.AudioDownloadInterrupted, match="cannot be confirmed to fit"):
        source.download_analysis_source(URL, VIDEO_ID, duration=37864, budget=proxy.ByteBudget(1000), download_full=full)
    timeline.assert_not_called()
    full.assert_not_called()


def test_small_full_fallback_still_available_when_no_partial_can_resume(direct, monkeypatch, tmp_path):
    failure = proxy.AudioDownloadInterrupted("Audio timed out", full_estimate_bytes=100, resumable=False)
    monkeypatch.setattr(source, "download_audio_for_job", Mock(side_effect=failure))
    timeline = Mock(return_value=tmp_path / "small-full.mp4")
    monkeypatch.setattr(source, "_full_timeline_via_sections", timeline)
    assert source.download_analysis_source(
        URL, VIDEO_ID, duration=60, budget=proxy.ByteBudget(1000), download_full=Mock(),
    ) == (tmp_path / "small-full.mp4", False)
    timeline.assert_called_once()


def test_realistic_long_source_timeout_estimate_blocks_timeline_before_media(direct, monkeypatch):
    info = {"duration": 37864, "formats": [
        {"format_id": "303", "protocol": "https", "vcodec": "avc1", "acodec": "none", "height": 1080, "tbr": 2500},
        {"format_id": "140", "protocol": "https", "vcodec": "none", "acodec": "aac", "ext": "m4a", "abr": 128},
    ]}
    monkeypatch.setattr(proxy, "extract_video_info", lambda *a, **kw: info)
    transfer = Mock(side_effect=TimeoutError("Read timed out (read timeout=0.13s)"))
    monkeypatch.setattr(proxy, "native_download_format", transfer)
    timeline = Mock(side_effect=AssertionError("Oversized full timeline must not start"))
    full = Mock(side_effect=AssertionError("Oversized full provider must not start"))
    monkeypatch.setattr(source, "_full_timeline_via_sections", timeline)
    with pytest.raises(proxy.AudioDownloadInterrupted, match="estimated at .* bytes with .* remaining") as error:
        source.download_analysis_source(URL, VIDEO_ID, duration=37864, budget=proxy.ByteBudget(2_000_000_000), download_full=full)
    assert error.value.full_estimate_bytes > 11_000_000_000
    assert transfer.call_count == 2
    timeline.assert_not_called()
    full.assert_not_called()


@pytest.fixture
def resumable_audio(direct, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    state = {"paths": [], "ranges": [], "http_failures": 1, "hls_failures": 1, "metadata": 0}
    tracks = {"/0.ts": b"a" * 32768, "/1.ts": b"b" * 32768, "/2.ts": b"c" * 32768}
    payload = b"one!" * 49152

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            state["paths"].append(self.path)
            state["ranges"].append(self.headers.get("Range"))
            if self.path == "/forbidden":
                self.send_error(403)
                return
            if self.path == "/1.ts" and state["hls_failures"]:
                state["hls_failures"] -= 1
                self.send_error(503)
                return
            data = payload if self.path == "/http" else tracks[self.path]
            header = self.headers.get("Range")
            offset = int(header.split("=", 1)[1].split("-", 1)[0]) if header else 0
            self.send_response(206 if header else 200)
            if header:
                self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - offset))
            self.end_headers()
            if self.path == "/http" and state["http_failures"]:
                state["http_failures"] -= 1
                self.wfile.write(data[offset:offset+65536])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(data[offset:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    mode = ["http"]
    def metadata(*a, **kw):
        state["metadata"] += 1
        http_available = mode[0] == "http" or (state.get("recover_http") and state["metadata"] > 1)
        formats = [{"format_id": "140", "protocol": "http", "url": base + ("/http" if http_available else "/forbidden"),
                    "vcodec": "none", "acodec": "aac", "ext": "m4a", "abr": 128, "filesize": len(payload)}]
        if mode[0] == "hls":
            formats.append({"format_id": "233", "protocol": "m3u8_native", "url": base + "/manifest.m3u8",
                            "vcodec": "none", "acodec": "aac", "ext": "mp4", "abr": 48,
                            "fragments": [{"url": base + name, "duration": 20} for name in tracks]})
        # Re-extraction may change only numeric representation, not the source.
        return {"duration": 60 if state["metadata"] == 1 else 60.0, "formats": formats}
    original = proxy.base_proxy_params
    monkeypatch.setattr(proxy, "base_proxy_params", lambda *a, **kw: dict(original(*a, **kw), fixup="never", noprogress=True))
    monkeypatch.setattr(proxy, "extract_video_info", metadata)
    monkeypatch.setattr(proxy, "contain_raw_aac", lambda path: path)
    monkeypatch.setattr(proxy, "probe_media_duration", lambda path: (
        path.stat().st_size / 32768 * 20 if str(path).endswith(".part") else 60.0
    ))
    try:
        yield state, payload, tracks, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_retry_resumes_with_range_and_charges_only_new_bytes(direct, resumable_audio, tmp_path):
    state, payload, _, _ = resumable_audio
    budget = proxy.ByteBudget(2_000_000)
    result = source.download_audio_for_job(URL, VIDEO_ID, budget=budget)
    assert result.read_bytes() == payload
    assert state["ranges"] == [None, "bytes=65536-"]
    assert state["metadata"] == 2
    assert budget.snapshot() == len(payload)
    assert not list(tmp_path.rglob("*.part*"))
    assert not list(tmp_path.rglob("*.ytdl"))


def test_hls_retry_reuses_completed_fragments_and_ytdlp_checkpoint(direct, resumable_audio, tmp_path):
    state, _, tracks, mode = resumable_audio
    mode[0] = "hls"
    budget = proxy.ByteBudget(2_000_000)
    result = source.download_audio_for_job(URL, VIDEO_ID, budget=budget)
    assert result.read_bytes() == b"".join(tracks.values())
    assert state["paths"].count("/0.ts") == 1
    assert state["paths"].count("/1.ts") == 2
    assert state["paths"].count("/2.ts") == 1
    assert state["metadata"] == 2
    assert budget.snapshot() == sum(map(len, tracks.values()))
    assert not list(tmp_path.rglob("*.part*"))
    assert not list(tmp_path.rglob("*.ytdl"))


def test_saved_hls_is_resumed_even_if_https_format_recovers(direct, resumable_audio, monkeypatch):
    state, _, tracks, mode = resumable_audio
    mode[0] = "hls"
    state["recover_http"] = True
    contain = Mock(side_effect=lambda path: path)
    monkeypatch.setattr(proxy, "contain_raw_aac", contain)
    result = source.download_audio_for_job(URL, VIDEO_ID, budget=proxy.ByteBudget(2_000_000))
    assert result.read_bytes() == b"".join(tracks.values())
    assert "/http" not in state["paths"]
    assert state["paths"].count("/0.ts") == 1
    contain.assert_called_once()


def test_interrupted_audio_remains_resumable_on_a_later_job(direct, resumable_audio, tmp_path):
    import json
    state, payload, _, _ = resumable_audio
    state["http_failures"] = 2
    first = proxy.ByteBudget(2_000_000)
    with pytest.raises(proxy.AudioDownloadInterrupted) as failure:
        source.download_audio_for_job(URL, VIDEO_ID, budget=first)
    assert failure.value.resumable is True
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.part")) == 131072
    manifest = tmp_path / "youtube-cache" / VIDEO_ID / "manifest.json"
    assert json.loads(manifest.read_text())["audio"] is None
    second = proxy.ByteBudget(2_000_000)
    result = source.download_audio_for_job(URL, VIDEO_ID, budget=second)
    assert result.read_bytes() == payload
    assert state["ranges"] == [None, "bytes=65536-", "bytes=131072-"]
    assert first.snapshot() == 131072 and second.snapshot() == len(payload) - 131072
    assert not list(tmp_path.rglob("*.part*"))
