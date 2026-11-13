/** Local Claude selector (selection: "external"): what an awaiting_selection task page should say. */

const SELECTOR_STALE_MS = 3 * 60_000;
/** Progress the local selector sets when it claims a task (backend SELECTION_CLAIMED_PROGRESS). */
const SELECTION_CLAIMED_PROGRESS = 55;

/**
 * working: claimed, heartbeat fresh. stalled: claimed, heartbeat stopped.
 * starting: just stopped for a selection. off: nobody claimed it within 3 minutes.
 */
export function claudeSelectorState(
  task: { progress?: number; updated_at: string }, now = Date.now(),
): "working" | "starting" | "off" | "stalled" {
  const claimed = (task.progress ?? 0) >= SELECTION_CLAIMED_PROGRESS;
  const fresh = now - Date.parse(task.updated_at) < SELECTOR_STALE_MS;
  if (claimed) return fresh ? "working" : "stalled";
  return fresh ? "starting" : "off";
}
