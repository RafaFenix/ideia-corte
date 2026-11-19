"""Framing is tied to an edit occurrence, independent of text/B-roll continuity."""
import json
import shutil
import subprocess

import pytest
from pydantic import ValidationError
from src.editor_document import EditDocument, effective_framing, visual_segments, continuous_segments, mapped_words, caption_key, render_document
from src.services.ai_edit_service import _subtract, apply_plan, EditPlan
from src.services.editor_service import upgrade_document

FRAME = {"fit": "cover", "zoom": 1, "x": 1, "y": .5}


def draft(**changes):
    return EditDocument.model_validate({"segments": [{"id": "a", "start": 0, "end": .4}, {"id": "b", "start": .4, "end": .8, "framing": FRAME}, {"id": "c", "start": .8, "end": 1.2}], **changes})


def test_legacy_defaults_and_complete_snapshot_validation():
    doc = draft(framing={"x": .2})
    assert doc.segments[0].framing is None
    assert effective_framing(doc, doc.segments[0]).x == .2
    assert effective_framing(doc, doc.segments[1]).x == 1
    assert EditDocument.model_validate(doc.model_dump()) == doc
    for bad in [{"x": 1}, {**FRAME, "aspect": "square"}, {**FRAME, "x": -1}, {**FRAME, "zoom": float("nan")}]:
        with pytest.raises(ValidationError):
            draft(segments=[{"id": "a", "start": 0, "end": 1, "framing": bad}])
    legacy = {"segments": [{"id": "a", "start": 0, "end": 1}], "framing": {"x": .2}}
    upgraded = upgrade_document(legacy, {}, {})
    assert upgraded["segments"][0]["framing"] is None
    assert upgrade_document(upgraded, {}, {}) == upgraded
    assert legacy["segments"][0].get("framing") is None


def test_visual_boundaries_do_not_change_caption_occurrences():
    doc = draft(words=[{"id": "w", "text": "hello", "start": .3, "end": .9}])
    assert len(visual_segments(doc)) == 3
    assert len(continuous_segments(doc.segments)) == 1
    assert len(mapped_words(doc)) == 1
    assert caption_key(doc, mapped_words(doc)) == '["a","w"]'
    assert [s.id for s in doc.segments] == ["a", "b", "c"]
    same = draft(segments=[{"id": "a", "start": 0, "end": .4}, {"id": "b", "start": .4, "end": .8}])
    assert len(visual_segments(same)) == 1


def test_ai_cuts_copy_segment_fields():
    [first, second] = _subtract([{"id": "a", "start": 0, "end": 2, "framing": FRAME}], [(.5, 1)])
    assert first["framing"] == second["framing"] == FRAME
    assert first["id"] != second["id"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize("audio,auto,repeated", [(True, False, False), (False, False, False), (True, True, False), (False, True, True)])
def test_real_hard_cuts_and_auto_manual_export(tmp_path, monkeypatch, audio, auto, repeated):
    import src.editor_document as module
    # Keep the real renderer and geometry, with a small 9:16 output for fast pixel checks.
    monkeypatch.setattr(module, "output_size", lambda *_: (180, 320))
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=360x180:r=30:d=1.2,drawbox=x=180:y=0:w=180:h=180:color=blue:t=fill"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1.2"]
    subprocess.run([*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "clean.mp4")], check=True, capture_output=True)
    if auto:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=yellow:s=180x320:r=30:d=1.2", "-c:v", "libx264", str(tmp_path / "layout.mp4")], check=True, capture_output=True)
    doc = draft(framing={"fit": "auto" if auto else "cover", "x": 0}, captions={"enabled": False})
    if repeated:
        doc = doc.model_copy(update={"segments": [doc.segments[2], doc.segments[1].model_copy(update={"start": 0, "end": .4}), doc.segments[0]]})
    render_document(tmp_path, "framing", doc, "tiktok")
    output = tmp_path / "export-framing.mp4"
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (180, 320)
    assert float(probe["format"]["duration"]) == pytest.approx(1.2, abs=.05)
    assert any(s["codec_type"] == "audio" for s in probe["streams"]) == audio
    colors = []
    for at in [.2, .6, 1.0]:
        pixel = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(output), "-frames:v", "1", "-vf", "crop=2:2:88:158,scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        colors.append(tuple(pixel[:3]))
    assert colors[1][2] > 200 and colors[1][0] < 40
    assert colors[0][0] > 200 and colors[2][0] > 200
    if auto:
        assert colors[0][1] > 200 and colors[2][1] > 200


async def test_ai_keep_ranges_is_guarded_and_global_settings_preserve_overrides():
    doc = draft()
    plan = EditPlan.model_validate({"summary": "Keep", "operations": [{"type": "keep_ranges", "ranges": [{"start": 0, "end": 1}]}, {"type": "set", "setting": "framing.x", "value": "0.2"}]})
    result, notes = await apply_plan(doc, plan, 1.2)
    assert result.segments == doc.segments
    assert any("Keep ranges skipped" in note for note in notes)
    assert result.framing.x == .2
    assert effective_framing(result, result.segments[1]).x == 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_mixed_sources_keep_broll_input_indices_and_continuous_dissolves(tmp_path, monkeypatch):
    import src.editor_document as module
    monkeypatch.setattr(module, "output_size", lambda *_: (180, 320))
    for name, color, size in [("clean", "red", "360x180"), ("layout", "yellow", "180x320"), ("stock", "blue", "180x320")]:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color={color}:s={size}:r=30:d=1.2", "-c:v", "libx264", str(tmp_path / f"{name}.mp4")], check=True, capture_output=True)
    stock = {"id": "stock", "start": .39, "end": 1, "footage": {"id": 1, "width": 180, "height": 320, "duration": 1.2, "fileUrl": "https://videos.pexels.com/1.mp4"}}
    doc = draft(framing={"fit": "auto"}, captions={"enabled": False}, broll=[stock])
    render_document(tmp_path, "stock", doc, "tiktok", {"1/180x320": tmp_path / "stock.mp4"})
    output = tmp_path / "export-stock.mp4"
    def pixel(at):
        return subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(output), "-frames:v", "1", "-vf", "crop=2:2:88:158,scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    assert pixel(.65)[2] > 200  # Stock input, not the yellow layout.
    assert pixel(1.1)[0] > 200 and pixel(1.1)[1] > 200
    # .39 -> .4 is a sub-frame continuation: the fade must survive that split.
    assert 0 < pixel(.43)[2] < 180


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_short_different_crops_do_not_accumulate_frame_rounding_drift(tmp_path, monkeypatch):
    import src.editor_document as module
    monkeypatch.setattr(module, "output_size", lambda *_: (180, 320))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=360x180:r=30:d=1,drawbox=x=180:y=0:w=180:h=180:color=blue:t=fill", "-c:v", "libx264", str(tmp_path / "clean.mp4")], check=True, capture_output=True)
    segments = [{"id": str(i), "start": i * .04, "end": (i + 1) * .04, "framing": {**FRAME, "x": 1 if i == 9 else 0, "zoom": 1 + i / 100}} for i in range(10)]
    doc = draft(segments=segments, captions={"enabled": False})
    render_document(tmp_path, "short", doc, "tiktok")
    rgb = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "0.36", "-i", str(tmp_path / "export-short.mp4"), "-frames:v", "1", "-vf", "crop=2:2:88:158,scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    assert rgb[2] > 200 and rgb[0] < 40  # The tenth run must still reach the output.
