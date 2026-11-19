import { claudeSelectorState } from "./claude-selector";

const at = (minutesAgo: number) => new Date(Date.UTC(2026, 10, 13, 12, 0) - minutesAgo * 60_000).toISOString();
const now = Date.UTC(2026, 10, 13, 12, 0);

it("tells a working selector from one that is off or stopped", () => {
  expect(claudeSelectorState({ progress: 55, updated_at: at(1) }, now)).toBe("working");
  expect(claudeSelectorState({ progress: 55, updated_at: at(5) }, now)).toBe("stalled");
  expect(claudeSelectorState({ progress: 50, updated_at: at(1) }, now)).toBe("starting");
  expect(claudeSelectorState({ progress: 50, updated_at: at(5) }, now)).toBe("off");
});
