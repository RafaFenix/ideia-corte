"""Carry download recovery notices from blocking providers to task progress."""

import asyncio
from contextvars import ContextVar
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar

from .utils.async_helpers import run_in_thread
from .download_cancellation import DownloadCancellation, DownloadCancelled, cancellation_scope


class DownloadRecovery(str, Enum):
    CONNECTION = "Retrying video download with another connection..."
    AUDIO = "Recovering video audio..."
    FULL_VIDEO = "Recovering the full video download..."
    BACKUP = "Using a backup video download option..."
    ANOTHER_BACKUP = "Trying another video download recovery option..."


_listener: ContextVar[Optional[Callable[[DownloadRecovery], None]]] = ContextVar(
    "download_recovery_listener", default=None,
)
T = TypeVar("T")
ProgressCallback = Callable[[int, str, str], Awaitable[None]]


def report_download_recovery(recovery: DownloadRecovery) -> None:
    listener = _listener.get()
    if listener is not None:
        listener(recovery)


async def run_download_in_thread(
    func: Callable[..., T], *args: Any,
    progress_callback: Optional[ProgressCallback] = None,
    progress: int = 10, should_cancel: Optional[Callable[[], Awaitable[bool]]] = None, **kwargs: Any,
) -> T:
    """Persist notices on the event loop while the provider runs in a thread.

    The listener belongs only to this operation. Callbacks finish before the
    next pipeline stage, and never use an async database session in a thread.
    """
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[DownloadRecovery] = asyncio.Queue()
    control = DownloadCancellation()
    closed = False

    def send(recovery: DownloadRecovery) -> None:
        if not closed:
            loop.call_soon_threadsafe(updates.put_nowait, recovery)

    def execute() -> T:
        token = _listener.set(send)
        try:
            with cancellation_scope(control):
                return func(*args, **kwargs)
        finally:
            _listener.reset(token)

    operation = asyncio.create_task(run_in_thread(execute))
    previous = None

    async def flush_notices():
        nonlocal previous
        while not updates.empty():
            recovery = updates.get_nowait()
            if progress_callback is not None and recovery != previous:
                await progress_callback(progress, recovery.value, "processing")
                previous = recovery

    try:
        while not operation.done():
            await asyncio.wait({operation}, timeout=.25)
            if should_cancel is not None and await should_cancel():
                control.stop()
            await flush_notices()
        await flush_notices()
        try:
            control.check()
            return operation.result()
        except DownloadCancelled as exc:
            raise RuntimeError("Task cancelled") from exc
    except BaseException:
        closed = True
        control.stop()
        # Never cancel the thread's awaiting Task: that would detach the writer.
        # Join it before returning, releasing media leases or cleaning its files.
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not operation.cancelled():
            operation.exception()
        raise
    finally:
        closed = True
