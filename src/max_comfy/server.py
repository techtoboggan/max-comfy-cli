"""Optional ComfyUI server lifecycle helper.

Most users will already have a ComfyUI server running. For users who want
max-comfy to manage one, ``Server`` spawns it as a subprocess, waits for
readiness, and tears it down on exit. The ComfyUI installation must be
available on disk; this module never imports ComfyUI itself.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .client import Client
from .exceptions import ServerError

log = logging.getLogger(__name__)


class Server:
    """Manages a ComfyUI subprocess.

    Example:
        >>> server = Server(comfyui_path="~/ComfyUI", port=8188)
        >>> server.start()
        >>> # ... do work ...
        >>> server.stop()

    Or as a context manager:
        >>> with Server(comfyui_path="~/ComfyUI") as srv:
        ...     client = Client(port=srv.port)
        ...     client.submit_and_wait(graph)
    """

    def __init__(
        self,
        comfyui_path: Path | str,
        host: str = "127.0.0.1",
        port: int = 8188,
        output_dir: Path | str | None = None,
        input_dir: Path | str | None = None,
        python_exe: str | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        log_file: Path | str | None = None,
        start_timeout: float = 60.0,
    ) -> None:
        self.comfyui_path = Path(comfyui_path).expanduser().resolve()
        if not self.comfyui_path.is_dir():
            raise ServerError(f"ComfyUI path does not exist: {self.comfyui_path}")
        if not (self.comfyui_path / "main.py").is_file():
            raise ServerError(
                f"ComfyUI path looks invalid (no main.py at {self.comfyui_path}). "
                "Point to the directory containing ComfyUI's main.py."
            )
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.input_dir = Path(input_dir).expanduser().resolve() if input_dir else None
        self.python_exe = python_exe or sys.executable
        self.extra_args = list(extra_args or [])
        self.env = dict(env) if env else None
        self.log_path = Path(log_file).expanduser() if log_file else None
        self.start_timeout = start_timeout
        self._process: subprocess.Popen[Any] | None = None
        self._log_handle: Any = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self.is_running():
            return
        cmd = self._build_command()
        log.info("Starting ComfyUI: %s", " ".join(cmd))
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("a", buffering=1)
            stdout: Any = self._log_handle
            stderr: Any = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.comfyui_path),
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as exc:
            raise ServerError(f"Failed to spawn ComfyUI: {exc}") from exc

        client = Client(host=self.host, port=self.port)
        try:
            client.wait_ready(timeout=self.start_timeout)
        except Exception:
            self.stop()
            raise
        log.info("ComfyUI is ready on %s:%d", self.host, self.port)

    def stop(self, timeout: float = 30.0) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("ComfyUI did not exit within %.0fs; killing", timeout)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                proc.wait(timeout=10)
        self._process = None
        if self._log_handle is not None:
            with contextlib.suppress(OSError):
                self._log_handle.close()
            self._log_handle = None

    def restart(self) -> None:
        self.stop()
        # Brief pause to let the OS reclaim the port.
        time.sleep(1.0)
        self.start()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_command(self) -> list[str]:
        cmd = [
            self.python_exe,
            "-u",
            "main.py",
            "--listen",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.output_dir:
            cmd += ["--output-directory", str(self.output_dir)]
        if self.input_dir:
            cmd += ["--input-directory", str(self.input_dir)]
        cmd += self.extra_args
        return cmd

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> Server:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
