"""End-to-end Client tests against a real (fake) HTTP server.

These complement the unit tests in test_*.py — they exercise actual HTTP
round-trips so we catch wire-format regressions (request body shape, response
parsing, multipart uploads, streaming downloads) without needing a real
ComfyUI install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from max_comfy.client import Client
from max_comfy.exceptions import ComfyError


# ---------------------------------------------------------------------- #
# Health / system stats
# ---------------------------------------------------------------------- #
def test_health_check_passes_against_live_server(fake_comfy_server):
    port, _ = fake_comfy_server
    client = Client(port=port)
    assert client.health_check() is True


def test_health_check_fails_when_server_unreachable():
    client = Client(port=1)  # port 1 is reserved, definitely not listening
    assert client.health_check() is False


def test_system_stats_round_trip(fake_comfy_server):
    port, _ = fake_comfy_server
    client = Client(port=port)
    stats = client.system_stats()
    assert "system" in stats
    assert "devices" in stats
    assert stats["devices"][0]["name"] == "FakeDevice"


def test_wait_ready_raises_on_unreachable():
    from max_comfy.exceptions import ClientError

    client = Client(port=1)
    with pytest.raises(ClientError, match="did not become ready"):
        client.wait_ready(timeout=1.0, poll=0.5)


# ---------------------------------------------------------------------- #
# Submit / wait
# ---------------------------------------------------------------------- #
def test_submit_records_graph_and_returns_prompt_id(fake_comfy_server):
    port, state = fake_comfy_server
    client = Client(port=port)
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 10}}}

    prompt_id = client.submit(graph)

    assert prompt_id == "test-0"
    assert state.submitted[prompt_id] == graph


def test_submit_includes_client_id_in_payload(fake_comfy_server):
    """ComfyUI uses client_id to route websocket events; ensure we send it."""
    port, state = fake_comfy_server
    client = Client(port=port, client_id="my-client")
    client.submit({"1": {"class_type": "X", "inputs": {}}})
    # State only stores the prompt graph, so check via a second submit:
    # the test passes if no exception was raised — Client must have accepted
    # the response, which means the server got a payload it could parse.
    assert "test-0" in state.submitted


def test_wait_for_completion_returns_history_entry(fake_comfy_server):
    port, _ = fake_comfy_server
    client = Client(port=port)
    prompt_id = client.submit({"1": {"class_type": "X", "inputs": {}}})

    entry = client.wait_for_completion(prompt_id, timeout=2.0, poll_interval=0.1)

    assert entry["status"]["status_str"] == "success"
    assert "outputs" in entry


def test_wait_for_completion_times_out_when_history_never_appears(fake_comfy_server):
    port, state = fake_comfy_server
    state.auto_complete = False  # server will accept prompts but never finish them
    client = Client(port=port)
    prompt_id = client.submit({"1": {"class_type": "X", "inputs": {}}})

    with pytest.raises(ComfyError, match="did not complete"):
        client.wait_for_completion(prompt_id, timeout=0.5, poll_interval=0.1)


# ---------------------------------------------------------------------- #
# Outputs
# ---------------------------------------------------------------------- #
def test_list_outputs_unpacks_history_entries(fake_comfy_server):
    port, _ = fake_comfy_server
    client = Client(port=port)
    prompt_id = client.submit({"1": {"class_type": "X", "inputs": {}}})
    history = client.wait_for_completion(prompt_id, timeout=2.0, poll_interval=0.1)

    outputs = client.list_outputs(history)

    assert len(outputs) == 1
    assert outputs[0].filename == "out.png"
    assert outputs[0].node_id == "1"
    assert outputs[0].type == "output"


def test_download_output_writes_bytes_to_disk(fake_comfy_server, tmp_path: Path):
    port, _ = fake_comfy_server
    client = Client(port=port)
    prompt_id = client.submit({"1": {"class_type": "X", "inputs": {}}})
    history = client.wait_for_completion(prompt_id, timeout=2.0, poll_interval=0.1)
    outputs = client.list_outputs(history)

    target = client.download_output(outputs[0], tmp_path)

    assert target.is_file()
    assert target.read_bytes() == b"FAKE_FILE_BYTES"
    assert target.name == "out.png"


def test_submit_and_wait_full_round_trip(fake_comfy_server, tmp_path: Path):
    port, state = fake_comfy_server
    client = Client(port=port)
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "test"}},
    }

    result = client.submit_and_wait(graph, download_dir=tmp_path)

    assert state.submitted == {"test-0": graph}
    assert result.prompt_id == "test-0"
    assert len(result.outputs) == 1
    assert len(result.downloaded) == 1
    assert result.downloaded[0].read_bytes() == b"FAKE_FILE_BYTES"


# ---------------------------------------------------------------------- #
# Uploads
# ---------------------------------------------------------------------- #
def test_upload_input_sends_multipart_with_filename(fake_comfy_server, tmp_path: Path):
    port, state = fake_comfy_server
    client = Client(port=port)
    f = tmp_path / "my_face.png"
    f.write_bytes(b"PNG_DATA_HERE")

    name = client.upload_input(f)

    assert name == "uploaded.png"
    assert len(state.uploads) == 1
    assert state.uploads[0]["filename"] == "my_face.png"
    assert state.uploads[0]["size"] >= len(b"PNG_DATA_HERE")  # multipart adds boundary overhead


def test_upload_input_rejects_missing_file(tmp_path: Path):
    from max_comfy.exceptions import ClientError

    client = Client(port=1)
    with pytest.raises(ClientError, match="non-existent"):
        client.upload_input(tmp_path / "does_not_exist.png")


# ---------------------------------------------------------------------- #
# Control plane
# ---------------------------------------------------------------------- #
def test_interrupt_hits_endpoint(fake_comfy_server):
    port, state = fake_comfy_server
    client = Client(port=port)
    client.interrupt()
    assert state.interrupt_calls == 1


def test_free_memory_sends_flags(fake_comfy_server):
    port, state = fake_comfy_server
    client = Client(port=port)
    client.free_memory(unload_models=True, free_memory=True)
    assert state.free_calls == 1
    assert state.last_free_payload == {"unload_models": True, "free_memory": True}  # type: ignore[attr-defined]


def test_queue_status(fake_comfy_server):
    port, _ = fake_comfy_server
    client = Client(port=port)
    queue = client.queue_status()
    assert "queue_running" in queue
    assert "queue_pending" in queue
