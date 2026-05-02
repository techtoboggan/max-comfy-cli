"""Shared pytest fixtures.

The :func:`fake_comfy_server` fixture spins up an in-process HTTP server that
mimics ComfyUI's REST endpoints (``/system_stats``, ``/prompt``,
``/history/<id>``, ``/view``, ``/upload/image``, ``/interrupt``, ``/free``).
End-to-end client tests use it to exercise the real HTTP wire format without
needing a real ComfyUI install. The fixture also exposes the recorded server
state (submitted graphs, uploaded files) so tests can assert on what hit the
wire.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest


@dataclass
class FakeComfyState:
    """Mutable state shared between handler instances and the test."""

    submitted: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: dict[str, dict[str, Any]] = field(default_factory=dict)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    free_calls: int = 0
    interrupt_calls: int = 0
    next_prompt_id: int = 0
    # Default behaviour: every queued prompt is auto-marked complete with a
    # single image output. Tests can override by assigning to ``auto_complete``.
    auto_complete: bool = True
    output_filename: str = "out.png"
    output_node_id: str = "1"
    file_bytes: bytes = b"FAKE_FILE_BYTES"


def _make_handler(state: FakeComfyState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any, **kwargs: Any) -> None:  # silence access log
            pass

        def do_GET(self) -> None:  # noqa: N802 — required by BaseHTTPRequestHandler
            url = urlparse(self.path)
            if url.path == "/system_stats":
                self._json(
                    {
                        "system": {"os": "test", "python_version": "3.x"},
                        "devices": [
                            {
                                "name": "FakeDevice",
                                "vram_total": 8 * 1024**3,
                                "vram_free": 7 * 1024**3,
                            }
                        ],
                    }
                )
                return
            if url.path.startswith("/history/"):
                prompt_id = url.path.split("/", 2)[-1]
                entry = state.history.get(prompt_id)
                self._json({prompt_id: entry} if entry is not None else {})
                return
            if url.path == "/view":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(state.file_bytes)))
                self.end_headers()
                self.wfile.write(state.file_bytes)
                return
            if url.path == "/queue":
                self._json({"queue_running": [], "queue_pending": []})
                return
            self._not_found()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            url = urlparse(self.path)

            if url.path == "/prompt":
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    return
                prompt_id = f"test-{state.next_prompt_id}"
                state.next_prompt_id += 1
                state.submitted[prompt_id] = data.get("prompt", {})
                if state.auto_complete:
                    state.history[prompt_id] = {
                        "outputs": {
                            state.output_node_id: {
                                "images": [
                                    {
                                        "filename": state.output_filename,
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                        "status": {"completed": True, "status_str": "success"},
                    }
                self._json({"prompt_id": prompt_id, "number": 0, "node_errors": {}})
                return

            if url.path == "/upload/image":
                # Roughly parse the multipart body to capture the filename.
                ctype = self.headers.get("Content-Type", "")
                filename = "unknown"
                if "multipart/form-data" in ctype:
                    try:
                        # Rough parser: just look for filename="..." in the headers section.
                        marker = b'filename="'
                        idx = body.find(marker)
                        if idx != -1:
                            end = body.find(b'"', idx + len(marker))
                            if end != -1:
                                filename = body[idx + len(marker) : end].decode("utf-8")
                    except Exception:  # noqa: BLE001 — never let parsing kill the test server
                        pass
                state.uploads.append({"filename": filename, "size": len(body)})
                self._json({"name": "uploaded.png", "subfolder": "", "type": "input"})
                return

            if url.path == "/interrupt":
                state.interrupt_calls += 1
                self._json({})
                return

            if url.path == "/free":
                state.free_calls += 1
                try:
                    state.last_free_payload = json.loads(body) if body else {}  # type: ignore[attr-defined]
                except json.JSONDecodeError:
                    state.last_free_payload = {}  # type: ignore[attr-defined]
                self._json({})
                return

            self._not_found()

        # ------------------------------------------------------------------ #
        def _json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _not_found(self) -> None:
            self.send_response(404)
            self.end_headers()

    return _Handler


@pytest.fixture
def fake_comfy_server():
    """Start an in-process fake ComfyUI HTTP server on a free port.

    Yields a tuple ``(port, state)`` — the test should connect a Client to
    ``port`` and assert against ``state.submitted`` / ``state.uploads`` etc.
    """
    state = FakeComfyState()
    handler_cls = _make_handler(state)
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# Convenience for tests that don't care about state assertions, only the URL.
@pytest.fixture
def fake_comfy_url(fake_comfy_server):
    port, _ = fake_comfy_server
    return f"http://127.0.0.1:{port}"
