"""Recognize legacy yt-dlp partials without matching transcripts or other videos."""

import re
from pathlib import Path


_PARTIAL = re.compile(r"(?P<id>[A-Za-z0-9_-]{11})\.(?:[^/]+\.)?(?:part(?:-Frag\d+)?|ytdl(?:\.tmp)?)")


def is_youtube_partial(name: str, video_id: str | None = None) -> bool:
    match = _PARTIAL.fullmatch(name)
    return bool(match and (video_id is None or match["id"] == video_id))


def cleanup_legacy_partials(directory: Path, video_id: str) -> None:
    for path in directory.glob(f"{video_id}.*"):
        if path.is_file() and is_youtube_partial(path.name, video_id):
            path.unlink(missing_ok=True)
