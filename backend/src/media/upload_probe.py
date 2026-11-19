"""Bounded metadata checks for uploaded media, including fragmented MP4.

Keep this separate from render probes: generation's legacy media path must not
change. Inputs are local uploads or short-lived URLs minted by MediaStorage.
"""

import json
import math
from pathlib import Path
import subprocess
import time


UPLOAD_PROBE_TIMEOUT_SECONDS = 15.0
UPLOAD_PREFLIGHT_TIMEOUT_SECONDS = 17.0


def _number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _probe(source: str | Path, deadline: float, *, tail: bool = False) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Upload metadata deadline exceeded")
    command = [
        "ffprobe", "-v", "error", "-rw_timeout", str(max(1, int(remaining * 1_000_000))),
        "-analyzeduration", "1000000", "-probesize", "1048576",
    ]
    if tail:
        # Seek beyond any admitted upload. The MOV demuxer uses its fragment
        # index (mfra when available) to land at the last random-access point,
        # then reads through the last fragment instead of trusting early moov
        # duration values. No decoding or media copy is needed.
        command += ["-read_intervals", "999999999%"]
    entries = "format=format_name,duration,start_time:stream=index,codec_type"
    if tail:
        entries += ":packet=stream_index,pts_time,dts_time,duration_time"
    command += ["-show_entries", entries, "-of", "json", str(source)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=remaining)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps ffprobe before raising. Keep signed
        # URLs and ffprobe's command line out of errors/logs.
        raise TimeoutError("Upload metadata deadline exceeded") from None
    if time.monotonic() >= deadline:
        raise TimeoutError("Upload metadata deadline exceeded")
    if result.returncode:
        raise ValueError("Unable to read upload metadata")
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise ValueError("Invalid upload metadata") from None
    if not isinstance(payload, dict):
        raise ValueError("Invalid upload metadata")
    return payload


def upload_duration(source: str | Path) -> float:
    """Return the complete uploaded length within one shared wall-clock budget.

    Ordinary containers keep their header duration. MP4/MOV also inspect the
    final packets: OBS/fMP4 headers can describe only the opening fragment.
    FFprobe's documented read_intervals and MOV fragment index support handle
    both local files and ranged object-storage reads.
    """
    deadline = time.monotonic() + UPLOAD_PROBE_TIMEOUT_SECONDS
    metadata = _probe(source, deadline)
    header = metadata.get("format") or {}
    duration = _number(header.get("duration"))
    if "mov" in str(header.get("format_name", "")).split(","):
        final = _probe(source, deadline, tail=True)
        streams = {
            stream["index"] for stream in final.get("streams", [])
            if stream.get("codec_type") in {"video", "audio"}
        }
        start = _number((final.get("format") or {}).get("start_time")) or 0.0
        ends = []
        for packet in final.get("packets", []):
            if packet.get("stream_index") not in streams:
                continue
            pts = _number(packet.get("pts_time"))
            if pts is None:
                pts = _number(packet.get("dts_time"))
            sample_duration = _number(packet.get("duration_time"))
            if pts is not None and sample_duration is not None and sample_duration > 0:
                ends.append(pts + sample_duration - start)
        # Don't silently accept a first-fragment header when the final
        # fragment could not be read. No task or minute charge exists yet.
        if not ends:
            raise ValueError("Unable to verify the final upload fragment")
        candidates = [*ends, _number((final.get("format") or {}).get("duration")), duration]
        duration = max(value for value in candidates if value is not None)
    if duration is None or duration <= 0:
        raise ValueError("Invalid upload duration")
    return duration
