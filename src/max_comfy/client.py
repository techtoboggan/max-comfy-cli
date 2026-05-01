"""ComfyUI HTTP client.

A thin, dependency-light wrapper over the ComfyUI REST API. Supports submitting
prompt graphs, polling for completion, and downloading produced files. No
ComfyUI Python imports are required at runtime — the client speaks only to a
running server over HTTP.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .exceptions import ClientError, ComfyError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputFile:
    """A file produced by ComfyUI for a single workflow node."""

    node_id: str
    filename: str
    subfolder: str
    type: str  # "output", "temp", "input"

    def query(self) -> dict[str, str]:
        return {"filename": self.filename, "subfolder": self.subfolder, "type": self.type}


@dataclass
class RunResult:
    """The outcome of a workflow submission."""

    prompt_id: str
    history: dict[str, Any]
    outputs: list[OutputFile] = field(default_factory=list)
    downloaded: list[Path] = field(default_factory=list)


class Client:
    """Thin client for the ComfyUI HTTP API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        client_id: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id or uuid.uuid4().hex
        self.request_timeout = request_timeout
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/system_stats", timeout=5.0)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, timeout: float = 60.0, poll: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health_check():
                return
            time.sleep(poll)
        raise ClientError(
            f"ComfyUI server at {self.base_url} did not become ready within {timeout:.0f}s"
        )

    def system_stats(self) -> dict[str, Any]:
        r = self._get("/system_stats")
        return r.json()

    # ------------------------------------------------------------------ #
    # Prompt submission
    # ------------------------------------------------------------------ #
    def submit(self, graph: dict[str, Any], extra_data: dict[str, Any] | None = None) -> str:
        """Queue a prompt graph and return its prompt_id."""
        payload: dict[str, Any] = {"prompt": graph, "client_id": self.client_id}
        if extra_data:
            payload["extra_data"] = extra_data
        r = self._post("/prompt", json=payload)
        try:
            data = r.json()
        except ValueError as exc:
            raise ClientError(f"ComfyUI returned non-JSON response: {r.text[:200]!r}") from exc
        if r.status_code != 200:
            raise ComfyError(
                f"Prompt rejected ({r.status_code}): {data}", details=data if isinstance(data, dict) else None
            )
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"Server response missing prompt_id: {data}")
        log.debug("Queued prompt %s", prompt_id)
        return prompt_id

    def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        r = self._get(f"/history/{prompt_id}")
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get(prompt_id)

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 7200.0,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        """Poll ``/history/<id>`` until the prompt finishes, then return its history entry."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = self.get_history(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(
                        f"ComfyUI returned an error for prompt {prompt_id}",
                        prompt_id=prompt_id,
                        details=status,
                    )
                if status.get("completed") or "outputs" in entry:
                    return entry
            time.sleep(poll_interval)
        self.interrupt(quiet=True)
        raise ComfyError(
            f"Prompt {prompt_id} did not complete within {timeout:.0f}s",
            prompt_id=prompt_id,
        )

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def list_outputs(self, history_entry: dict[str, Any]) -> list[OutputFile]:
        """Flatten a history entry's per-node outputs into a list of OutputFile."""
        files: list[OutputFile] = []
        outputs = history_entry.get("outputs") or {}
        for node_id, payload in outputs.items():
            for kind in ("images", "gifs", "videos", "audio", "files"):
                for item in payload.get(kind) or []:
                    files.append(
                        OutputFile(
                            node_id=str(node_id),
                            filename=item.get("filename", ""),
                            subfolder=item.get("subfolder", ""),
                            type=item.get("type", "output"),
                        )
                    )
        return files

    def download_output(self, output: OutputFile, dest: Path) -> Path:
        """Download an output file to `dest` (file path or directory)."""
        params = output.query()
        url = f"{self.base_url}/view?{urllib.parse.urlencode(params)}"
        r = self._session.get(url, timeout=self.request_timeout, stream=True)
        if r.status_code != 200:
            raise ClientError(f"Failed to download {output.filename}: HTTP {r.status_code}")
        if dest.is_dir() or (not dest.exists() and not dest.suffix):
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / output.filename
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            target = dest
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        return target

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def submit_and_wait(
        self,
        graph: dict[str, Any],
        timeout: float = 7200.0,
        poll_interval: float = 1.0,
        download_dir: Path | None = None,
    ) -> RunResult:
        """Submit a graph, wait for it, and (optionally) download all outputs."""
        prompt_id = self.submit(graph)
        history = self.wait_for_completion(prompt_id, timeout=timeout, poll_interval=poll_interval)
        outputs = self.list_outputs(history)
        downloaded: list[Path] = []
        if download_dir is not None:
            download_dir.mkdir(parents=True, exist_ok=True)
            for f in outputs:
                downloaded.append(self.download_output(f, download_dir))
        return RunResult(
            prompt_id=prompt_id, history=history, outputs=outputs, downloaded=downloaded
        )

    # ------------------------------------------------------------------ #
    # Inputs (uploads)
    # ------------------------------------------------------------------ #
    def upload_input(
        self,
        local_path: Path | str,
        subfolder: str = "",
        overwrite: bool = True,
        kind: str = "input",
    ) -> str:
        """Upload a file to ComfyUI's input directory and return the filename.

        Use this to send local frames or reference images into ComfyUI before
        building a graph that references them with ``LoadImage``. The returned
        filename may differ from the local one (ComfyUI appends a counter to
        avoid collisions when ``overwrite=False``).
        """
        path = Path(local_path)
        if not path.is_file():
            raise ClientError(f"Cannot upload non-existent file: {path}")
        with path.open("rb") as f:
            files = {"image": (path.name, f, "application/octet-stream")}
            data = {"type": kind, "overwrite": "true" if overwrite else "false"}
            if subfolder:
                data["subfolder"] = subfolder
            try:
                r = self._session.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data=data,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                raise ClientError(f"Upload failed: {exc}") from exc
        if r.status_code != 200:
            raise ClientError(f"Upload failed: HTTP {r.status_code}: {r.text[:200]}")
        try:
            payload = r.json()
        except ValueError:
            return path.name
        name = payload.get("name") or path.name
        if subfolder:
            return f"{subfolder}/{name}"
        return name

    def interrupt(self, quiet: bool = False) -> None:
        """Cancel the currently-running prompt, if any."""
        try:
            self._post("/interrupt")
        except (requests.RequestException, ClientError):
            if not quiet:
                raise

    def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask ComfyUI to drop loaded models / clear VRAM."""
        try:
            self._post("/free", json={"unload_models": unload_models, "free_memory": free_memory})
        except (requests.RequestException, ClientError):
            log.warning("free_memory request failed; continuing")

    def queue_status(self) -> dict[str, Any]:
        return self._get("/queue").json()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get(self, path: str) -> requests.Response:
        try:
            return self._session.get(f"{self.base_url}{path}", timeout=self.request_timeout)
        except requests.RequestException as exc:
            raise ClientError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, json: dict[str, Any] | None = None) -> requests.Response:
        try:
            return self._session.post(
                f"{self.base_url}{path}", json=json, timeout=self.request_timeout
            )
        except requests.RequestException as exc:
            raise ClientError(f"POST {path} failed: {exc}") from exc

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
