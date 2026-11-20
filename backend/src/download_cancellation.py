"""Per-download cancellation for native HTTP readers and yt-dlp subprocesses."""

from contextlib import contextmanager
from contextvars import ContextVar
import os
import signal
import socket
import subprocess
import threading


class DownloadCancelled(BaseException):
    """Internal stop signal; provider retries must never catch it as a failure."""


_current = ContextVar("download_cancellation", default=None)
_install_lock = threading.Lock()
_installed = False


class DownloadCancellation:
    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.processes = set()
        self.sockets = set()

    def check(self):
        if self.event.is_set():
            raise DownloadCancelled("Task cancelled")

    def stop(self):
        self.event.set()
        with self.lock:
            for connection in tuple(self.sockets):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            for process in tuple(self.processes):
                if process.poll() is None:
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    except ProcessLookupError:
                        pass


@contextmanager
def cancellation_scope(control):
    token = _current.set(control)
    try:
        control.check()
        yield
        control.check()
    finally:
        _current.reset(token)


def check_download_cancelled():
    control = _current.get()
    if control is not None:
        control.check()


def run_local_process(command, *, capture_output=False, text=False, timeout=None, **kwargs):
    """Local mux/probe work within a download must stop with that download too."""
    if _current.get() is None:
        return subprocess.run(command, capture_output=capture_output, text=text, timeout=timeout, **kwargs)
    _install_process_tracking()
    check_download_cancelled()
    from yt_dlp.downloader import external

    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check = kwargs.pop("check", False)
    stdout, stderr, returncode = external.Popen.run(command, text=text, timeout=timeout, **kwargs)
    check_download_cancelled()
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _install_process_tracking():
    """Track only subprocesses launched inside the calling thread's scope.

    yt-dlp has no cancellation callback for its ffmpeg downloader/postprocessors.
    Install its Popen adapter once; unrelated jobs retain the original behavior.
    Each tracked child owns a process group, so cancellation cannot kill another job.
    """
    global _installed
    with _install_lock:
        if _installed:
            return
        from yt_dlp.downloader import external
        from yt_dlp.postprocessor import ffmpeg

        original = external.Popen

        class TrackedPopen(original):
            def __init__(self, *args, **kwargs):
                self.download_control = _current.get()
                if self.download_control is not None:
                    self.download_control.check()
                    if os.name == "posix":
                        kwargs["start_new_session"] = True
                super().__init__(*args, **kwargs)
                if self.download_control is not None:
                    with self.download_control.lock:
                        self.download_control.processes.add(self)
                    if self.download_control.event.is_set():
                        self.download_control.stop()

            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    if self.download_control is not None:
                        with self.download_control.lock:
                            self.download_control.processes.discard(self)
                        self.download_control.check()

        external.Popen = ffmpeg.Popen = TrackedPopen
        _installed = True


class CancellableResponse:
    def __init__(self, response, control):
        self.response, self.control = response, control
        # requests/urllib wrap sockets in several layers. Shutdown the owned
        # socket, rather than closing a buffered reader that may hold its lock
        # while a peer trickles bytes indefinitely below the read size.
        self.sockets = set()
        pending, seen = [response], set()
        while pending and len(seen) < 24:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if isinstance(item, socket.socket):
                self.sockets.add(item)
                continue
            for name in ("fp", "raw", "_fp", "_sock", "sock", "_connection"):
                value = getattr(item, name, None)
                if value is not None:
                    pending.append(value)
        with control.lock:
            control.sockets.update(self.sockets)
        if control.event.is_set():
            control.stop()

    def __getattr__(self, name):
        return getattr(self.response, name)

    def read(self, size=-1):
        self.control.check()
        try:
            if size is None or size < 0:
                chunks = []
                while chunk := self.read(65_536):
                    chunks.append(chunk)
                return b"".join(chunks)
            data = self.response.read(min(size, 65_536))
            self.control.check()
            return data
        except BaseException:
            self.close()
            self.control.check()
            raise

    read1 = read

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.control.lock:
            self.control.sockets.difference_update(self.sockets)
        self.response.close()


@contextmanager
def managed_ytdlp_client(options):
    import yt_dlp
    from yt_dlp.networking import Request

    control = _current.get()
    if control is not None:
        _install_process_tracking()
        control.check()
    with yt_dlp.YoutubeDL(options) as client:
        if control is not None and hasattr(client, "urlopen"):
            original = client.urlopen

            def open_response(request):
                control.check()
                if isinstance(request, str):
                    request = Request(request)
                # A blocked socket cannot keep the cancelled thread alive forever.
                request.extensions["timeout"] = min(5.0, float(request.extensions.get("timeout") or 30))
                try:
                    response = CancellableResponse(original(request), control)
                    audio_clock = options.get("_youtube_audio_clock")
                    if audio_clock:
                        # Connection setup stays bounded above. Once headers
                        # arrive, audio may wait for its configured idle window;
                        # cancellation still interrupts the registered socket.
                        for connection in response.sockets:
                            try:
                                connection.settimeout(audio_clock.idle_seconds)
                            except OSError:
                                pass
                    return response
                except BaseException:
                    control.check()
                    raise

            client.urlopen = open_response
        try:
            yield client
            check_download_cancelled()
        except BaseException:
            check_download_cancelled()
            raise
