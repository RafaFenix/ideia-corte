"""External clip selection: an MCP client picks the clips instead of the backend LLM.

A task created with ``selection: "external"`` stops after transcription in
``awaiting_selection``. The client reads the transcript (``transcript_page``) and
submits segments (``validate_selection``); the accepted ones become the task
checkpoint's ``analysis`` and the normal render runs on them.
"""

from typing import Any, Dict, List, Optional, Tuple

from .ai import (
    TranscriptSegment,
    ViralityAnalysis,
    _extract_transcript_text,
    _format_minutes_label,
    _parse_transcript_lines,
    _parse_transcript_spans,
    _parse_transcript_timestamp_seconds,
    _overlap_ratio,
    _segment_bounds_seconds,
    _validate_segments,
    CANDIDATE_DEDUPE_OVERLAP_RATIO,
    max_final_clips_for_duration,
)
from .clip_duration import clip_duration

SELECTION_MODE_KEY = "selection_mode"
EXTERNAL = "external"
AWAITING_SELECTION = "awaiting_selection"
MAX_PAGE_LINES = 500
SCORE_KEYS = ("hook", "engagement", "value", "shareability")


class AwaitingSelection(Exception):
    """Raised by the pipeline once the transcript is saved and the client must pick clips."""


def is_external(metadata: Any) -> bool:
    return isinstance(metadata, dict) and metadata.get(SELECTION_MODE_KEY) == EXTERNAL


def selection_limits(transcript: str, clip_length: str, max_clips: int) -> Dict[str, Any]:
    lines = _parse_transcript_lines(transcript)
    duration = max((line["end_seconds"] for line in lines), default=0)
    bounds = clip_duration(clip_length)
    return {
        "duration_seconds": duration,
        "min_seconds": bounds.minimum,
        "max_seconds": bounds.maximum,
        "ideal_min_seconds": bounds.ideal_minimum,
        "ideal_max_seconds": bounds.ideal_maximum,
        "max_clips": max_final_clips_for_duration(duration, max_clips),
    }


def transcript_page(
    transcript: str,
    transcript_data: Optional[Dict[str, Any]],
    offset: int = 0,
    limit: int = 200,
    include_words: bool = False,
) -> Dict[str, Any]:
    """One page of the timestamped lines the LLM normally reads, plus optional word timings."""
    lines = _parse_transcript_lines(transcript)
    offset = max(0, offset)
    limit = max(1, min(MAX_PAGE_LINES, limit))
    page = lines[offset : offset + limit]
    result: Dict[str, Any] = {
        "total_lines": len(lines),
        "next_offset": offset + limit if offset + limit < len(lines) else None,
        "lines": [
            {
                "i": offset + index,
                "start": line["start_time"],
                "end": line["end_time"],
                "start_seconds": line["start_seconds"],
                "end_seconds": line["end_seconds"],
                "text": line["text"],
            }
            for index, line in enumerate(page)
        ],
    }
    if include_words and page:
        # Line labels are whole seconds; include words starting within the last one.
        low, high = page[0]["start_seconds"], page[-1]["end_seconds"] + 1
        words = (transcript_data or {}).get("words") or []
        result["words"] = [
            {"t": word.get("text"), "s": word.get("start"), "e": word.get("end")}
            for word in words
            if isinstance(word, dict) and low * 1000 <= (word.get("start") or 0) < high * 1000
        ]
    return result


def _seconds(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("timestamp must be MM:SS, HH:MM:SS or seconds")
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str) and value.strip():
        text = value.strip()
        return int(round(float(text))) if ":" not in text else _parse_transcript_timestamp_seconds(text)
    raise ValueError("timestamp must be MM:SS, HH:MM:SS or seconds")


def _to_segment(item: Dict[str, Any], spans: List[Dict[str, Any]]) -> TranscriptSegment:
    start, end = _seconds(item.get("start")), _seconds(item.get("end"))
    if end <= start:
        raise ValueError("end must be after start")
    scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
    values = {key: max(0, min(25, int(scores.get(key) or 0))) for key in SCORE_KEYS}
    return TranscriptSegment(
        start_time=_format_minutes_label(start),
        end_time=_format_minutes_label(end),
        text=_extract_transcript_text(spans, start, end),
        relevance_score=1.0,
        reasoning=str(item.get("reasoning") or "Selected by the user's assistant.")[:500],
        virality=ViralityAnalysis(
            hook_score=values["hook"],
            engagement_score=values["engagement"],
            value_score=values["value"],
            shareability_score=values["shareability"],
            total_score=sum(values.values()),
        ),
        hook_title=item.get("hook_title") or None,
    )


def validate_selection(
    items: List[Dict[str, Any]], transcript: str, clip_length: str, max_clips: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Ground submitted segments like model output. Returns (segments, accepted, rejected).

    ``segments`` keep the submitted order (it is the ranking) and are shaped like
    ``analysis.most_relevant_segments`` in the task checkpoint.
    """
    lines = _parse_transcript_lines(transcript)
    spans = _parse_transcript_spans(transcript)
    limit = selection_limits(transcript, clip_length, max_clips)["max_clips"]
    bounds = clip_duration(clip_length)
    taken: List[Tuple[int, int]] = []
    segments: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "segment must be an object"})
            continue
        try:
            segment = _to_segment(item, spans)
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        requested = (segment.start_time, segment.end_time)
        validated = _validate_segments([segment], lines, spans, clip_length)
        if not validated:
            rejected.append({
                "index": index,
                "reason": (
                    "not usable: needs at least 3 transcribed words, bounds within 8s of a "
                    f"transcript line and {bounds.minimum}-{bounds.maximum}s duration"
                ),
            })
            continue
        segment = validated[0]
        span = _segment_bounds_seconds(segment)
        # Submission order is the ranking, so an earlier pick wins an overlap.
        if any(_overlap_ratio(span, other) > CANDIDATE_DEDUPE_OVERLAP_RATIO for other in taken):
            rejected.append({"index": index, "reason": "overlaps an earlier segment"})
            continue
        if len(segments) >= limit:
            rejected.append({"index": index, "reason": f"over max_clips ({limit})"})
            continue
        taken.append(span)
        payload = segment.model_dump()
        if item.get("respect_bounds", True) is not False:
            # Keeps _refine_segment_openings off the client's chosen in/out points.
            payload["opening_reviewed"] = True
        segments.append(payload)
        accepted.append({
            "index": index,
            "start": segment.start_time,
            "end": segment.end_time,
            "snapped": (segment.start_time, segment.end_time) != requested,
            "hook_title": segment.hook_title,
        })
    return segments, accepted, rejected
