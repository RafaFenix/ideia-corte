import { cleanup, render, screen } from "@testing-library/react";
import { StatusBadge } from "./status-badge";

afterEach(cleanup);

it("labels a task waiting for an external clip selection", () => {
  render(<StatusBadge status="awaiting_selection" />);
  expect(screen.getByText("Awaiting selection")).toBeInTheDocument();
});
