import { describe, expect, it, vi } from "vitest";
import { draft } from "./document.test-fixture";
import { captionKey, captionGroups, mappedWords, mappedBroll, renderSegments, effectiveFraming, framingSource, normalizeFraming, setFraming, splitSegment, visualSegments, locate, type EditDocument } from "./document";
import { drawEditorFrame } from "./render";

const personalized = () => setFraming(splitSegment(draft, 0, 1), draft.segments[0].id, "segment", { x: 0, zoom: 1.5 });
describe("segment framing", () => {
  it("loads legacy documents without changing their global crop", () => {
    const legacy = { ...draft, framing: { ...draft.framing, x: .2 } };
    const normalized = normalizeFraming(legacy);
    expect(normalizeFraming(normalized)).toEqual(normalized);
    expect(normalized.segments[0].framing).toBeNull();
    expect(effectiveFraming(normalized, normalized.segments[0])).toEqual(legacy.framing);
  });
  it("uses complete snapshots, global aspect and independent occurrences", () => {
    const doc = personalized(), original = structuredClone(doc);
    const split = splitSegment(doc, 0, .5);
    expect(split.segments[0].framing).toEqual(split.segments[1].framing);
    expect(split.segments[0].framing).not.toBe(split.segments[1].framing);
    const repeated = { ...doc, segments: [{ ...doc.segments[0], id: "repeat" }, ...doc.segments] };
    const changed = setFraming(repeated, "repeat", "segment", { x: 1 });
    expect(changed.segments[1].framing?.x).toBe(0);
    const square: EditDocument = { ...changed, framing: { ...changed.framing, aspect: "square", zoom: 2 } };
    expect(effectiveFraming(square, square.segments[1])).toMatchObject({ aspect: "square", zoom: 1.5 });
    expect(doc).toEqual(original);
  });
  it("applies to the whole clip atomically and restores inheritance", () => {
    const doc = personalized();
    const all = setFraming(doc, doc.segments[0].id, "clip", { x: .8 });
    expect(all.segments.every(s => s.framing === null)).toBe(true);
    expect(all.framing.x).toBe(.8);
    const inherited = { ...doc, segments: doc.segments.map(s => ({ ...s, framing: null })) };
    expect(effectiveFraming(inherited, inherited.segments[0])).toEqual(doc.framing);
  });
  it("coalesces only equal visual runs while keeping caption keys and tiny B-roll continuations", () => {
    const base = { ...draft, segments: [{ id: "a", start: 0, end: 1.01 }, { id: "b", start: 1.01, end: 3 }] };
    const doc = setFraming(base, "b", "segment", { x: 1 });
    expect(visualSegments(doc)).toHaveLength(2);
    expect(visualSegments(base)).toHaveLength(1);
    expect(captionKey(doc, captionGroups(mappedWords(doc), 4)[0])).toBe(captionKey(base, captionGroups(mappedWords(base), 4)[0]));
    const broll = [{ id: "stock", start: 1, end: 2, offset: 0, query: "", footage: { provider: "pexels" as const, id: 1, width: 320, height: 180, duration: 3, fileUrl: "https://videos.pexels.com/1.mp4", pageUrl: "", thumbnail: "", credit: "" } }];
    const pieces = mappedBroll({ ...doc, broll, segments: renderSegments(doc.segments) });
    expect(pieces).toHaveLength(1);
    expect(pieces[0]).toMatchObject({ start: 1, end: 2, fadeIn: true, fadeOut: true });
    expect(locate(doc, 1.01).index).toBe(1);
  });
  it("selects auto layouts only for vertical output with a layout asset", () => {
    expect(framingSource({ ...draft.framing, fit: "auto" }, true)).toBe("layout");
    expect(framingSource({ ...draft.framing, fit: "auto", aspect: "square" }, true)).toBe("clean");
    expect(framingSource({ ...draft.framing, fit: "auto" })).toBe("clean");
  });
  it("draws the crop of the output occurrence on each side of a hard cut", () => {
    const doc = setFraming(personalized(), personalized().segments[0].id, "segment", { x: 0 });
    const context = { canvas: { width: 108, height: 192 }, clearRect: vi.fn(), fillRect: vi.fn(), save: vi.fn(), restore: vi.fn() } as unknown as CanvasRenderingContext2D;
    const draw = vi.fn();
    drawEditorFrame(context, doc, 640, 360, .5, draw, undefined, null, "none");
    const left = draw.mock.calls[0][0];
    drawEditorFrame(context, doc, 640, 360, 1, draw, undefined, null, "none");
    expect(left.x).toBeCloseTo(0);
    expect(draw.mock.calls[1][0].x).toBeLessThan(0);
  });
});
