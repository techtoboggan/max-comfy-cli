"""Unit tests for the batch loader and progress tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from max_comfy.exceptions import WorkflowError
from max_comfy.runner import (  # noqa: PLC2701 — testing internals on purpose
    Batch,
    JobResult,
    _Progress,
)


def test_batch_load_full_shape(tmp_path: Path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({
        "workflow": "wf",
        "output_dir": "./out",
        "defaults": {"steps": 30},
        "jobs": [
            {"id": "a", "params": {"prompt": "x", "seed": 1}},
            {"id": "b", "prompt": "y", "seed": 2},  # flat shape too
            {"prompt": "z"},  # auto-generated id
        ],
    }))
    batch = Batch.load(path)
    assert batch.workflow == "wf"
    assert batch.output_dir == Path("./out")
    assert batch.defaults == {"steps": 30}
    assert len(batch.jobs) == 3

    a, b, c = batch.jobs
    assert a.id == "a"
    assert a.params == {"prompt": "x", "seed": 1}
    assert b.id == "b"
    assert b.params == {"prompt": "y", "seed": 2}
    assert c.id == "0002"
    assert c.params == {"prompt": "z"}


def test_batch_load_array_form(tmp_path: Path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([
        {"prompt": "x"},
        {"prompt": "y"},
    ]))
    batch = Batch.load(path)
    assert len(batch.jobs) == 2
    assert batch.workflow is None
    assert batch.jobs[0].params == {"prompt": "x"}


def test_batch_load_missing_file(tmp_path: Path):
    with pytest.raises(WorkflowError):
        Batch.load(tmp_path / "nope.json")


def test_batch_load_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(WorkflowError):
        Batch.load(path)


def test_batch_load_per_job_workflow_override(tmp_path: Path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({
        "workflow": "default",
        "jobs": [
            {"id": "a", "prompt": "x"},
            {"id": "b", "workflow": "other", "prompt": "y"},
        ],
    }))
    batch = Batch.load(path)
    assert batch.jobs[0].workflow is None  # use default
    assert batch.jobs[1].workflow == "other"


def test_progress_records_completed_only(tmp_path: Path):
    progress_path = tmp_path / ".progress.json"
    p = _Progress(progress_path)

    p.record(JobResult(job_id="a", success=True, files=["/x.png"]))
    p.record(JobResult(job_id="b", success=False, error="boom"))

    p2 = _Progress(progress_path)
    assert p2.completed == {"a"}
    assert len(p2.results) == 2  # both recorded, but only successes count toward completed


def test_progress_atomic_write(tmp_path: Path):
    """A crash between record() and successful save should not leave a corrupt file."""
    progress_path = tmp_path / ".progress.json"
    p = _Progress(progress_path)
    p.record(JobResult(job_id="a", success=True))
    # The .tmp file should be gone (replaced atomically).
    tmp_files = list(tmp_path.glob(".progress.json.tmp"))
    assert tmp_files == []
    assert progress_path.is_file()
    data = json.loads(progress_path.read_text())
    assert data["completed"] == ["a"]
