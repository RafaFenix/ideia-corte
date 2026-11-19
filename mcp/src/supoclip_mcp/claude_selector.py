"""Local Claude selector: the user's Claude subscription picks the clips.

Runs on the host, next to a self-hosted SupoClip. Tasks created with
``selection: "external"`` stop in ``awaiting_selection``; this loop claims the
oldest one, runs ``claude -p`` with only the supoclip MCP tools, and checks
that a selection went in. One task at a time (file lock).

    uv run --directory mcp supoclip-claude-selector [--user-id ID | --email ADDR]

Without either, the only admin user of the compose Postgres is used.

Auth (same modes as the MCP client): ``SUPOCLIP_API_KEY``, or
``SUPOCLIP_USER_ID`` + ``SUPOCLIP_AUTH_SECRET``. Without the secret it is read
from ``BACKEND_AUTH_SECRET`` in the repository's ``.env``. Credentials reach the
MCP server only through a temporary 0600 config file, never argv or logs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .client import SupoClipClient, SupoClipError
from .config import Settings, load_settings

AWAITING = "awaiting_selection"
CLAIMED_PROGRESS = 55
STALE_SECONDS = 180
HEARTBEAT_SECONDS = 60
REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path.home() / ".supoclip"
SELECTOR_TOOLS = ("get_transcript", "submit_selection", "get_task")
ALLOWED_TOOLS = ",".join(f"mcp__supoclip__supoclip_{name}" for name in SELECTOR_TOOLS)
AUTH_HINTS = ("login", "log in", "authenticat", "credit", "usage limit", "rate limit", "subscription")

log = logging.getLogger("claude-selector")

SYSTEM_PROMPT = """Você é um editor sênior de vídeos curtos (Shorts, Reels, TikTok) e escolhe os cortes de um vídeo longo.
Você só tem três ferramentas: supoclip_get_transcript, supoclip_submit_selection e supoclip_get_task.

Fluxo obrigatório:
1. Leia a transcrição INTEIRA com supoclip_get_transcript(task_id, limit=500), seguindo next_offset até ser null. Não use include_words.
   Em vídeos muito longos, anote os candidatos de cada página em poucas palavras (início, fim, ideia) em vez de repetir o texto.
2. Use os "limits" da resposta: durações entre min_seconds e max_seconds, mirando 25–50 s (ideal_min_seconds–ideal_max_seconds).
   Escolha exatamente limits.max_clips cortes; menos só se não houver momentos bons o suficiente.
3. Envie UMA vez com supoclip_submit_selection, do melhor para o pior, usando start/end de linhas da transcrição
   (start de uma linha, end de outra). Cortes não podem se sobrepor.
4. Se a resposta tiver "rejected" mas status "queued", pare: os aceitos já vão renderizar. Se vier erro 422 (nada aceito),
   corrija os limites e envie de novo uma única vez.

O que é um bom corte (critérios do SupoClip):
- Gancho nos primeiros 1–3 segundos: uma afirmação forte, pergunta, número ou contraste que faz quem rola o feed parar.
  As primeiras frases precisam fazer sentido sozinhas; não comece em conjunção ("Mas", "E aí", "Então") nem em resposta solta.
- Uma ideia completa e independente: quem não viu o resto do vídeo entende o contexto.
- Payoff: termina com a conclusão, a virada, a piada ou a lição — nunca no meio do raciocínio.
- Específico e concreto vale mais que genérico; histórias, números, opiniões fortes e conselhos acionáveis rendem mais.
- Evite introduções, despedidas, patrocínios, pedidos de inscrição e trechos que dependem de algo visual não descrito.

Para cada corte envie:
- hook_title: 3–9 palavras em português do Brasil (ou no idioma da transcrição, se não for português), fiel ao trecho,
  sem hashtags, emojis ou aspas.
- reasoning: uma frase curta.
- scores: {hook, engagement, value, shareability}, inteiros de 0 a 25, honestos.

Responda no final com uma linha: quantos cortes foram aceitos e quantos rejeitados."""


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def read_env_file(path: Path, key: str) -> Optional[str]:
    """One value from a dotenv file, without exporting anything else."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def lookup_user_id(email: Optional[str] = None) -> Optional[str]:
    """User id from the compose Postgres: by email, or the only admin when no email is given.

    The email is bound as a psql variable, never interpolated into the SQL.
    """
    query = (
        "SELECT id FROM users WHERE lower(email) = lower(:'email');\n" if email
        else "SELECT id FROM users WHERE is_admin;\n"
    )
    command = ["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "supoclip", "-d", "supoclip", "-tA"]
    if email:
        command += ["-v", f"email={email}"]
    result = subprocess.run(command, input=query, capture_output=True, text=True, cwd=REPO_ROOT, timeout=30)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return rows[0] if result.returncode == 0 and len(rows) == 1 else None


def resolve_settings(base: Settings, user_id: Optional[str], env_file: Path) -> Settings:
    if base.api_key:
        return base
    user_id = user_id or base.user_id
    if not user_id:
        raise SystemExit("Set SUPOCLIP_USER_ID (or pass --user-id / --email), or SUPOCLIP_API_KEY.")
    secret = base.auth_secret or read_env_file(env_file, "BACKEND_AUTH_SECRET")
    if not secret:
        raise SystemExit(f"SUPOCLIP_AUTH_SECRET is not set and BACKEND_AUTH_SECRET was not found in {env_file}.")
    return replace(base, user_id=user_id, auth_secret=secret)


def mcp_server_env(settings: Settings) -> Dict[str, str]:
    env = {
        "SUPOCLIP_API_URL": settings.api_url,
        "SUPOCLIP_MCP_TRANSPORT": "stdio",
        # Only the three tools the selector uses: the other schemas would cost context every turn.
        "SUPOCLIP_MCP_TOOLS": ",".join(f"supoclip_{name}" for name in SELECTOR_TOOLS),
    }
    if settings.api_key:
        env["SUPOCLIP_API_KEY"] = settings.api_key
    else:
        env["SUPOCLIP_USER_ID"] = settings.user_id or ""
        env["SUPOCLIP_AUTH_SECRET"] = settings.auth_secret or ""
    return env


@contextlib.contextmanager
def mcp_config_file(settings: Settings):
    """A 0600 MCP config holding the credentials; removed when the run ends."""
    config = {
        "mcpServers": {
            "supoclip": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "supoclip_mcp.server"],
                "env": mcp_server_env(settings),
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix="supoclip-mcp-", suffix=".json")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def claude_command(claude: str, task_id: str, config_path: str, model: str) -> List[str]:
    """``claude -p`` with nothing but the selection prompt and three supoclip tools.

    --setting-sources "" skips user/project/local settings, CLAUDE.md, hooks and plugins;
    --system-prompt replaces the default prompt; --tools "" drops every built-in tool;
    --strict-mcp-config ignores every other MCP server (claude.ai connectors included).
    --bare would go further but never reads the subscription login, so it is not used.
    """
    return [
        claude, "-p", f"Escolha os cortes da task {task_id}.",
        "--model", model,
        "--output-format", "json",
        "--no-session-persistence",
        "--setting-sources", "",
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", "",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config", config_path,
        "--allowedTools", ALLOWED_TOOLS,
        "--permission-mode", "dontAsk",
    ]


def claude_env() -> Dict[str, str]:
    """The CLI's environment without SupoClip credentials or an API key.

    Without ANTHROPIC_API_KEY the CLI uses the subscription login, not API billing.
    """
    return {
        key: value for key, value in os.environ.items()
        if not key.startswith("SUPOCLIP_") and key not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    }


# --------------------------------------------------------------------------- #
# Task choice
# --------------------------------------------------------------------------- #
def _age_seconds(task: Dict[str, Any], now: datetime) -> float:
    raw = str(task.get("updated_at") or "")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (now - stamp).total_seconds()


def pick_task(tasks: List[Dict[str, Any]], now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Oldest unclaimed awaiting task, or one whose selector stopped sending heartbeats."""
    now = now or datetime.now(timezone.utc)
    waiting = [task for task in tasks if task.get("status") == AWAITING]
    waiting.sort(key=lambda task: str(task.get("created_at") or ""))
    for task in waiting:
        claimed = int(task.get("progress") or 0) >= CLAIMED_PROGRESS
        if not claimed or _age_seconds(task, now) > STALE_SECONDS:
            return task
    return None


def failure_message(result: Dict[str, Any], timed_out: bool) -> Tuple[str, bool]:
    """(message for the task page, whether retrying could help)."""
    text = str(result.get("result") or result.get("error") or "")
    if any(hint in text.lower() for hint in AUTH_HINTS):
        return (
            "Claude CLI sem login ou sem cota. Rode `claude` no terminal, faça login e clique em Tentar novamente.",
            False,
        )
    if timed_out:
        return "Claude não terminou a seleção a tempo. Clique em Tentar novamente.", True
    return "Claude não conseguiu escolher cortes válidos. Clique em Tentar novamente.", True


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class Selector:
    def __init__(self, client: SupoClipClient, settings: Settings, *, claude: str, model: str,
                 timeout: int, attempts: int = 2):
        self.client = client
        self.settings = settings
        self.claude = claude
        self.model = model
        self.timeout = timeout
        self.attempts = attempts

    async def claim(self, task_id: str) -> bool:
        try:
            await self.client.request("POST", f"/tasks/{task_id}/selection/claim")
            return True
        except SupoClipError as exc:
            log.info("task %s: claim refused (%s)", task_id, str(exc)[:120])
            return False

    async def _heartbeat(self, task_id: str) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await self.claim(task_id)

    async def run_claude(self, task_id: str) -> Tuple[Dict[str, Any], bool]:
        """(parsed JSON result, timed out)."""
        with mcp_config_file(self.settings) as config_path:
            process = await asyncio.create_subprocess_exec(
                *claude_command(self.claude, task_id, config_path, self.model),
                stdin=asyncio.subprocess.DEVNULL,  # -p would read our stdin into the prompt
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(), env=claude_env(), start_new_session=True,
            )
            heartbeat = asyncio.create_task(self._heartbeat(task_id))
            try:
                stdout, _stderr = await asyncio.wait_for(process.communicate(), self.timeout)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                await process.wait()
                return {}, True
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                raise
            finally:
                heartbeat.cancel()
        try:
            return json.loads(stdout.decode("utf-8", "replace") or "{}"), False
        except ValueError:
            return {"error": stdout.decode("utf-8", "replace")[-300:]}, False

    async def still_waiting(self, task_id: str) -> bool:
        task = await self.client.request("GET", f"/tasks/{task_id}")
        return task.get("status") == AWAITING

    async def handle(self, task_id: str) -> str:
        if not await self.claim(task_id):
            return "skipped"
        message = ""
        for attempt in range(1, self.attempts + 1):
            started = asyncio.get_running_loop().time()
            result, timed_out = await self.run_claude(task_id)
            log.info(
                "task %s: attempt %d %s in %.0fs (turns=%s, cost_usd=%s)",
                task_id, attempt, "timed out" if timed_out else ("error" if result.get("is_error") else "done"),
                asyncio.get_running_loop().time() - started, result.get("num_turns"), result.get("total_cost_usd"),
            )
            if not await self.still_waiting(task_id):
                return "selected"
            message, retry = failure_message(result, timed_out)
            if not retry:
                break
            await self.claim(task_id)
        with contextlib.suppress(SupoClipError):
            await self.client.request("POST", f"/tasks/{task_id}/selection/fail", json_body={"message": message})
        log.warning("task %s: failed (%s)", task_id, message)
        return "failed"

    async def poll_once(self) -> Optional[str]:
        data = await self.client.request("GET", "/tasks/", params={"limit": 50})
        task = pick_task(data.get("tasks") or [])
        if task is None:
            return None
        return await self.handle(str(task["id"]))


@contextlib.contextmanager
def single_instance(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"Another Claude selector is already running ({lock_path}).")
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


async def run_forever(selector: Selector, poll_seconds: int) -> None:
    log.info("Claude selector started; polling every %ss (Ctrl+C to stop)", poll_seconds)
    while True:
        try:
            outcome = await selector.poll_once()
            if outcome:
                continue  # look for the next waiting task right away
        except SupoClipError as exc:
            log.warning("backend unavailable: %s", str(exc)[:200])
        await asyncio.sleep(poll_seconds)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-id", help="SupoClip user id (default: SUPOCLIP_USER_ID)")
    parser.add_argument("--email", help="Look the user id up by email (default: the only admin) in the compose Postgres")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--model", default=os.getenv("CLAUDE_SELECTOR_MODEL", "sonnet"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("CLAUDE_SELECTOR_TIMEOUT", "900")))
    parser.add_argument("--poll", type=int, default=int(os.getenv("CLAUDE_SELECTOR_POLL", "15")))
    parser.add_argument("--once", action="store_true", help="Handle at most one task and exit")
    args = parser.parse_args(argv)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(STATE_DIR / "claude-selector.log")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    claude = shutil.which("claude")
    if not claude:
        raise SystemExit("Claude Code CLI not found on PATH (install it and run `claude` once to log in).")

    base = load_settings()
    if not os.getenv("SUPOCLIP_API_URL"):
        base = replace(base, api_url="http://localhost:8000")
    user_id = args.user_id
    if not user_id and not base.api_key and not base.user_id:
        user_id = lookup_user_id(args.email)
        if not user_id:
            raise SystemExit(
                "Could not find exactly one user (by --email, or the single admin) in the compose Postgres. "
                "Pass --user-id or set SUPOCLIP_USER_ID."
            )
    settings = resolve_settings(base, user_id, Path(args.env_file))
    selector = Selector(SupoClipClient(settings), settings, claude=claude, model=args.model, timeout=args.timeout)

    with single_instance(STATE_DIR / "claude-selector.lock"):
        try:
            if args.once:
                log.info("result: %s", asyncio.run(selector.poll_once()) or "no task waiting")
            else:
                asyncio.run(run_forever(selector, args.poll))
        except KeyboardInterrupt:
            log.info("Claude selector stopped")


if __name__ == "__main__":
    main()
