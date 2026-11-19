# Claude selector (your Claude subscription picks the clips)

On a self-hosted SupoClip (or as an admin), the create screen offers
**Settings → Selection → Claude (subscription)**. The task then downloads and
transcribes the video and stops in `awaiting_selection`. A small process on
your computer, the Claude selector, takes it from there: it runs the Claude
Code CLI with your own login, lets Claude read the transcript and send the
clips, and the server renders them as usual (9:16, captions, hook title). The
server needs no LLM key for this; B-roll is skipped.

## Start and stop

Requirements: the Docker stack running, [uv](https://docs.astral.sh/uv/), and the
Claude Code CLI on `PATH`, logged in once (`claude`, then `/login`).

```bash
uv run --directory mcp supoclip-claude-selector        # Ctrl+C to stop
uv run --directory mcp supoclip-claude-selector --once # handle one waiting task and exit
```

Run it from the repository root, in any terminal (a Maestri canvas terminal works
too). Only one selector runs at a time (`~/.supoclip/claude-selector.lock`); it
handles one task at a time, oldest first, polling every 15 s.

## Who it acts as

| Setting | Default |
|---|---|
| `SUPOCLIP_USER_ID` / `--user-id` | the only admin in the compose Postgres (or `--email you@example.com`) |
| `SUPOCLIP_AUTH_SECRET` | `BACKEND_AUTH_SECRET` read from the repository `.env` (`--env-file`) |
| `SUPOCLIP_API_KEY` | optional alternative to the two above |
| `SUPOCLIP_API_URL` | `http://localhost:8000` |
| `CLAUDE_SELECTOR_MODEL` / `--model` | `sonnet` |
| `CLAUDE_SELECTOR_TIMEOUT` / `--timeout` | `900` seconds per attempt |
| `CLAUDE_SELECTOR_POLL` / `--poll` | `15` seconds |

The user id and secret reach the MCP server only through a temporary MCP config
file (mode 0600, deleted after each run), never `claude`'s arguments or the log.
The CLI runs without `SUPOCLIP_*` variables and without `ANTHROPIC_API_KEY`, so
it uses your subscription login rather than API billing.

## What Claude sees

`claude -p` runs with a fixed selection prompt (criteria adapted from
`backend/src/ai.py`, hook titles in the transcript's language, 25-50 s clips,
`limits.max_clips` of them) and is kept to the minimum:

- `--system-prompt` replaces Claude Code's default prompt;
- `--setting-sources ""` skips user/project/local settings, so no `CLAUDE.md`,
  hooks, plugins or permissions from your setup;
- `--tools ""` removes every built-in tool; `--disable-slash-commands` drops skills;
- `--strict-mcp-config` loads only the supoclip MCP server (stdio), and
  `SUPOCLIP_MCP_TOOLS` makes that server expose only `supoclip_get_transcript`,
  `supoclip_submit_selection` and `supoclip_get_task` (also the `--allowedTools` list);
- the working directory is the system temp directory.

Measured on a trivial prompt: about 0.7k input tokens instead of about 43k with the
defaults. Each turn with the three tools costs about 3.4k plus the transcript
pages (about 20k tokens per hour of video). `--bare` would trim further but never
reads the subscription login, so it is not used.

## When something goes wrong

- Claude sends nothing, a 422, or runs past the timeout: one more attempt, then
  the task fails with a clear message. **Retry** brings it back to
  `awaiting_selection` without transcribing again.
- The CLI is logged out or out of quota: the task fails right away saying so.
- The selector is not running: the task page says so after 3 minutes, with the
  command above. If it stops mid-task (no heartbeat for 3 minutes), the next
  selector run picks the task up again.

Log: `~/.supoclip/claude-selector.log` (task id, attempt, duration, turns, cost;
never the transcript or credentials).
