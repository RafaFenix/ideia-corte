"""Direct transport reuses audio, sections, cache and bounded full recovery."""

import copy
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from pathlib import Path

import pytest

from src import youtube_proxy as transport
from src import youtube_source as source
from src import youtube_utils as youtube
from src.config import Config, set_config_override
from src.services import video_service as service

VIDEO_ID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
PROXY = "http://user-session-{session}:pass@proxy.example:823"
FORMATS = [
    {"format_id": "140", "protocol": "https", "vcodec": "none", "acodec": "aac",
     "abr": 128, "ext": "m4a", "url": "https://rr1---sn-example.googlevideo.com/audio"},
    {"format_id": "audio", "protocol": "http_dash_segments", "vcodec": "none",
     "acodec": "aac", "ext": "m4a", "abr": 128, "url": "https://rr1---sn-example.googlevideo.com/a",
     "fragments": [{"url": f"https://rr1---sn-example.googlevideo.com/a{i}", "duration": 10} for i in range(6)]},
    {"format_id": "video", "protocol": "http_dash_segments", "vcodec": "avc1",
     "acodec": "none", "height": 720, "ext": "mp4", "url": "https://rr1---sn-example.googlevideo.com/v",
     "fragments": [{"url": f"https://rr1---sn-example.googlevideo.com/v{i}", "duration": 10} for i in range(6)]},
]


@pytest.fixture
def direct(tmp_path, monkeypatch):
    config = Config()
    config.temp_dir = str(tmp_path)
    config.youtube_proxy_url = ""
    config.youtube_sections_direct = True
    config.youtube_pot_provider_url = ""
    config.youtube_proxy_country = ""
    config.youtube_source_cache = True
    config.youtube_source_cache_ttl_seconds = 0
    config.youtube_section_padding_seconds = 0
    config.youtube_download_max_bytes = 1000
    config.media_storage = "local"
    config.youtube_download_provider = "yt_dlp"
    config.apify_api_token = ""
    config.rapidapi_youtube_api_key = ""
    set_config_override(config)
    monkeypatch.setattr(source, "_storage", lambda: None)
    yield config
    set_config_override(None)


@pytest.fixture
def native(monkeypatch):
    calls = []

    class YDL:
        live = False

        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def urlopen(self, _request):
            raise AssertionError("No remote playlist is needed for timed DASH fixtures")

        def extract_info(self, url, download=False):
            assert url == URL and not download
            return {"id": VIDEO_ID, "duration": 60, "formats": copy.deepcopy(FORMATS)}

        def write(self, ext):
            dest = Path(self.options["outtmpl"].replace("%(ext)s", ext))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.with_suffix(dest.suffix + ".part").write_bytes(b"12345678")
            for hook in self.options["progress_hooks"]:
                hook({"status": "downloading", "filename": str(dest), "downloaded_bytes": 8})
            dest.with_suffix(dest.suffix + ".part").replace(dest)
            return dest

        def process_info(self, info):
            calls.append(("native", self.options, copy.deepcopy(info)))
            assert self.options["external_downloader"] == {"default": "native"}
            assert "download_ranges" not in self.options
            self.write(info["ext"])

        def download(self, urls):
            assert urls == [URL]
            calls.append(("full", self.options, None))
            self.options["match_filter"]({"is_live": self.live})
            self.write("mp4")

    monkeypatch.setattr("yt_dlp.YoutubeDL", YDL)

    def mux(_video, _audio, dest, **_kwargs):
        dest.write_bytes(b"local mux")

    monkeypatch.setattr(transport, "_mux_local_window", mux)
    return calls, YDL


def test_env_defaults_off_and_parses_opt_in(monkeypatch):
    monkeypatch.delenv("YOUTUBE_SECTIONS_DIRECT", raising=False)
    assert Config().youtube_sections_direct is False
    monkeypatch.setenv("YOUTUBE_SECTIONS_DIRECT", "true")
    assert Config().youtube_sections_direct is True
    monkeypatch.setenv("YOUTUBE_SECTIONS_DIRECT", "false")
    assert Config().youtube_sections_direct is False


def test_disabled_keeps_legacy_full_download(direct, monkeypatch, tmp_path):
    direct.youtube_sections_direct = False
    full = tmp_path / "legacy.mp4"
    full.write_bytes(b"legacy")
    calls = []
    monkeypatch.setattr(youtube, "_download_youtube_video_with_ytdlp",
                        lambda *args: calls.append(args) or full)
    monkeypatch.setattr(youtube, "download_full_with_proxy",
                        lambda *a, **kw: pytest.fail("Disabled flag must keep legacy path"))
    assert not transport.sections_enabled(direct)
    assert youtube.download_youtube_video(URL) == full
    assert len(calls) == 1
    with pytest.raises(transport.YouTubeDownloadError, match="requires YOUTUBE_PROXY_URL"):
        source.download_audio_for_job(URL, VIDEO_ID)


@pytest.mark.parametrize("proxy", ["", PROXY])
def test_audio_and_sections_use_shared_cache_and_transport(direct, native, monkeypatch, caplog, proxy):
    direct.youtube_proxy_url = proxy
    calls, _ = native
    caplog.set_level(logging.INFO)
    budget = transport.ByteBudget(1000)
    audio, audio_first = source.download_analysis_source(
        URL, VIDEO_ID, budget=budget, duration=60,
        download_full=lambda: pytest.fail("Healthy audio must not fetch a full file"),
    )
    assert audio_first and audio.is_file()
    assert audio.parent == Path(direct.temp_dir) / "youtube-cache" / VIDEO_ID

    def assemble(slices, dest):
        assert slices[0][1:] == (20.0, 30.0)
        dest.write_bytes(b"assembled")
        mapping = source.build_time_map([(20.0, 30.0)])
        source.save_time_map(dest, mapping)
        return mapping

    monkeypatch.setattr(source, "assemble_timeline", assemble)
    segments = [{"start_time": 20, "end_time": 30}]
    rendered = source.materialize_clip_source(URL, VIDEO_ID, segments, budget=budget, duration=60)
    assert rendered.is_file()
    assert [item[2]["format_id"] for item in calls] == ["140", "video", "audio"]
    for _, options, info in calls:
        assert bool(options["proxy"]) is bool(proxy)
        assert "section_start" not in info and "section_end" not in info
    assert len(calls[1][2]["fragments"]) < 6
    before = budget.snapshot()
    assert source.download_audio_for_job(URL, VIDEO_ID, budget=budget) == audio
    source.materialize_clip_source(URL, VIDEO_ID, segments, budget=budget, duration=60)
    assert budget.snapshot() == before == 24
    assert len(calls) == 3
    manifest = json.loads((audio.parent / "manifest.json").read_text())
    assert manifest["audio"] == audio.name and len(manifest["ranges"]) == 1
    providers = {record.youtube_provider for record in caplog.records if hasattr(record, "youtube_provider")}
    assert ("yt_dlp_proxy" if proxy else "yt_dlp") in providers


def test_proxy_options_and_provider_order_unchanged_when_flag_enabled(direct, monkeypatch):
    direct.youtube_proxy_url = PROXY
    monkeypatch.setattr(transport.time, "monotonic", lambda: 100.0)
    def options():
        params = transport.download_options("task12345678", VIDEO_ID, transport.ByteBudget(100))
        return {k: v for k, v in params.items() if not callable(v) and k != "progress_hooks"}
    direct.youtube_sections_direct = False
    old = options()
    order = youtube._provider_order(direct, skip_proxy=False)
    direct.youtube_sections_direct = True
    assert options() == old
    assert youtube._provider_order(direct, skip_proxy=False) == order == [
        "yt_dlp_proxy", "rapidapi", "apify", "yt_dlp",
    ]
    assert youtube._provider_order(direct, skip_proxy=True) == ["rapidapi", "apify", "yt_dlp"]
    assert transport.sections_enabled(direct) and not transport.direct_sections_enabled(direct)


def test_sections_stop_at_shared_audio_plus_sections_budget(direct, native):
    budget = transport.ByteBudget(20)
    source.download_audio_for_job(URL, VIDEO_ID, budget=budget)
    with pytest.raises(transport.DownloadBudgetExceeded, match="20 byte job limit"):
        source.materialize_clip_source(
            URL, VIDEO_ID, [{"start_time": 20, "end_time": 30}], budget=budget, duration=60,
        )
    assert budget.snapshot() == 24
    assert not list(Path(direct.temp_dir).rglob("*.part"))
    assert not list(Path(direct.temp_dir).rglob(".parts-*"))
    assert not json.loads((Path(direct.temp_dir) / "youtube-cache" / VIDEO_ID / "manifest.json").read_text())["ranges"]


def test_full_fallback_cannot_escape_budget_or_leave_partial_files(direct, native, monkeypatch):
    calls, _ = native
    direct.youtube_download_provider = "apify"
    direct.apify_api_token = "configured-but-not-used"
    for name in ("_download_youtube_video_with_ytdlp", "download_youtube_video_with_apify", "download_video_via_rapidapi"):
        monkeypatch.setattr(youtube, name, lambda *a, **kw: pytest.fail("Unbounded fallback called"))
    budget = transport.ByteBudget(12)
    budget.add(8)  # Audio has already used part of the job allowance.
    with pytest.raises(transport.DownloadBudgetExceeded, match="12 byte job limit"):
        youtube.download_youtube_video(URL, budget=budget, skip_proxy=True)
    assert [item[0] for item in calls] == ["full"]
    assert calls[0][1]["proxy"] == ""
    assert "[protocol=https]" in calls[0][1]["format"]
    assert not list(Path(direct.temp_dir).rglob("*.part"))
    assert not list(Path(direct.temp_dir).glob("proxy-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("playlist", [
    "#EXTM3U\n#EXTINF:60,\npart.ts\n",
    '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="key"\n#EXTINF:60,\npart.ts\n#EXT-X-ENDLIST\n',
])
async def test_unsupported_hls_falls_back_to_bounded_full_file(direct, native, monkeypatch, playlist):
    calls, ydl = native
    formats = copy.deepcopy(FORMATS)
    for fmt in formats:
        fmt.pop("fragments", None)
        fmt["protocol"] = "m3u8_native"
    monkeypatch.setattr(ydl, "extract_info", lambda *a, **kw: {"duration": 60, "formats": formats})
    class Response:
        url = "https://rr1---sn-example.googlevideo.com/playlist.m3u8"
        def read(self, size):
            return playlist.encode()
        def close(self):
            pass
    monkeypatch.setattr(ydl, "urlopen", lambda *a: Response())
    # Any remote cache/metadata operation remains isolated from real services.
    monkeypatch.setattr(service.MediaStorage, "restore_youtube_metadata", lambda *a: None)
    budget = transport.ByteBudget(1000)
    full = await service.VideoService.download_video(URL, ranges=[(20, 30)], budget=budget)
    assert full.is_file()
    assert [item[0] for item in calls] == ["full"]
    assert budget.snapshot() == len(playlist.encode()) + 8


def test_ongoing_live_fails_clearly_before_media_transfer(direct, native):
    calls, ydl = native
    ydl.live = True
    budget = transport.ByteBudget(100)
    with pytest.raises(transport.YouTubeDownloadError, match="ongoing live stream"):
        youtube.download_youtube_video(URL, budget=budget)
    assert budget.snapshot() == 0
    assert len(calls) == 1
    assert not list(Path(direct.temp_dir).glob("proxy-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in,proxy,audio_first", [(False, "", False), (True, "", True), (True, PROXY, True)])
async def test_processing_routes_opt_in_to_audio_first(direct, monkeypatch, tmp_path, opt_in, proxy, audio_first):
    direct.youtube_sections_direct = opt_in
    direct.youtube_proxy_url = proxy
    path = tmp_path / "analysis.m4a"
    path.write_bytes(b"audio")
    calls = []
    async def info(*a, **kw):
        return {"duration": 60}
    async def full(*a, **kw):
        calls.append("full")
        return path
    def analysis(*a, **kw):
        calls.append("audio")
        return path, True
    async def prepare(path):
        return path
    class ReachedTranscription(RuntimeError):
        pass
    async def transcribe(*a, **kw):
        raise ReachedTranscription()
    monkeypatch.setattr(service, "async_get_youtube_video_info", info)
    monkeypatch.setattr(service, "cached_full_media", lambda *a: None)
    monkeypatch.setattr(service, "download_analysis_source", analysis)
    monkeypatch.setattr(service.VideoService, "download_video", full)
    monkeypatch.setattr(service.VideoService, "prepare_source", prepare)
    monkeypatch.setattr(service.VideoService, "_get_file_duration", lambda *a: 60)
    monkeypatch.setattr(service.VideoService, "generate_transcript", transcribe)
    with pytest.raises(ReachedTranscription):
        await service.VideoService.process_video_complete(URL, "youtube")
    assert calls == (["audio"] if audio_first else ["full"])


def test_real_native_direct_transfer_stops_at_budget(direct, monkeypatch, tmp_path):
    payload = b"x" * (2 * 1024 * 1024)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def metadata(*_args, **_kwargs):
        return {"duration": 60, "formats": [
            {"format_id": "140", "protocol": "http", "vcodec": "none", "acodec": "aac",
             "abr": 128, "ext": "m4a", "url": f"http://127.0.0.1:{server.server_port}/audio"},
        ]}
    monkeypatch.setattr(transport, "extract_video_info", metadata)
    budget = transport.ByteBudget(128 * 1024)
    try:
        with pytest.raises(transport.DownloadBudgetExceeded, match="byte job limit"):
            transport.download_audio(URL, VIDEO_ID, tmp_path / "audio", budget=budget)
        assert budget.limit < budget.snapshot() <= budget.limit + 64 * 1024
        assert not list(tmp_path.glob("audio.*"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
