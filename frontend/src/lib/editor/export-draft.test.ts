import { afterEach, describe, expect, it, vi } from "vitest";
import type { EditorState } from "./document";
import { draft } from "./document.test-fixture";
import { exportDraft } from "./export-draft";

const runtime = vi.hoisted(() => ({ inputs: [] as { dispose: ReturnType<typeof vi.fn> }[], failLayout: false, abort: null as AbortController | null, cancel: vi.fn(async () => {}) }));
vi.mock("./render", () => ({ loadEditorFont: vi.fn().mockResolvedValue(undefined), drawEditorFrame: vi.fn() }));
vi.mock("mediabunny", () => {
  const track = { canDecode: async () => true };
  return {
    canEncodeVideo: async () => true, canEncodeAudio: async () => true,
    ALL_FORMATS: [],
    BlobSource: class {}, BufferTarget: class {}, Mp4OutputFormat: class {},
    Input: class {
      index = runtime.inputs.length;
      dispose = vi.fn();
      constructor() { runtime.inputs.push(this); }
      async getPrimaryVideoTrack() {
        if (this.index === 1) {
          if (runtime.failLayout) throw new Error("Layout decoder failed");
          runtime.abort?.abort();
        }
        return track;
      }
      async getPrimaryAudioTrack() { return null; }
    },
    Output: class {
      cancel = runtime.cancel;
      addVideoTrack() {}
      async start() {}
    },
    CanvasSource: class { close() {} },
    VideoSampleSink: class { async *samples() {} },
  };
});
afterEach(() => vi.unstubAllGlobals());
describe("mixed framing export resource lifecycle", () => {
  for (const failure of ["decode", "abort"] as const) it(`disposes both inputs and cancels output on ${failure}`, async () => {
    runtime.inputs = [];
    runtime.cancel.mockClear();
    runtime.failLayout = failure === "decode";
    const control = new AbortController();
    runtime.abort = failure === "abort" ? control : null;
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => new Blob() })));
    vi.stubGlobal("document", { createElement: () => ({ width: 0, height: 0, getContext: () => ({}) }) });
    const doc = { ...draft, framing: { ...draft.framing, fit: "auto" as const } };
    await expect(exportDraft("/editor/media/clean.mp4", doc, { hasLayout: true, width: 640, height: 360, hasAudio: false } as EditorState, "tiktok", control.signal, vi.fn())).rejects.toThrow();
    expect(runtime.inputs).toHaveLength(2);
    for (const input of runtime.inputs) expect(input.dispose).toHaveBeenCalledOnce();
    expect(runtime.cancel).toHaveBeenCalledOnce();
  });
});
