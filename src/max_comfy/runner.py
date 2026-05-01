"""Batch job runner.

Reads a jobs file (JSON), iterates the jobs, and executes each via a
:class:`~max_comfy.workflows.Workflow`. Supports resume via a ``.progress.json``
file inside the output directory: jobs whose ``id`` is already recorded as
completed are skipped.

Jobs file shape:

.. code-block:: json

    {
      "workflow": "sdxl-txt2img",
      "output_dir": "./output",
      "defaults": {"steps": 30, "cfg": 7.0},
      "jobs": [
        {"id": "sunset", "params": {"prompt": "a sunset over mountains", "seed": 42}},
        {"id": "forest", "params": {"prompt": "a misty forest", "seed": 100}}
      ]
    }

Per-job ``id`` is optional — if absent, an index-based id is generated.
A per-job ``workflow`` field overrides the top-level workflow.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .client import Client
from .config import Config
from .exceptions import MaxComfyError, WorkflowError
from .server import Server
from .workflows import RunContext, Workflow, WorkflowRegistry
from .workflows import registry as default_registry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------- #
@dataclass
class Job:
    """A single batch job."""

    id: str
    params: dict[str, Any]
    workflow: str | None = None  # overrides batch-level workflow


@dataclass
class JobResult:
    job_id: str
    success: bool
    files: list[str] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """A loaded jobs file."""

    workflow: str | None
    output_dir: Path
    defaults: dict[str, Any]
    jobs: list[Job]
    raw_config: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Batch:
        path = Path(path)
        if not path.is_file():
            raise WorkflowError(f"Jobs file not found: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"Invalid JSON in {path}: {exc}") from exc

        if isinstance(data, list):
            data = {"jobs": data}
        elif not isinstance(data, dict):
            raise WorkflowError(f"Jobs file root must be an object or array, got {type(data).__name__}")

        raw_jobs = data.get("jobs") or []
        if not isinstance(raw_jobs, list):
            raise WorkflowError("`jobs` must be an array")

        jobs: list[Job] = []
        for index, item in enumerate(raw_jobs):
            if not isinstance(item, dict):
                raise WorkflowError(f"Job at index {index} must be an object")
            job_id = str(item.get("id") or f"{index:04d}")
            # Allow either a flat job (params at top level) or a nested {params: {...}} job.
            if "params" in item and isinstance(item["params"], dict):
                params = dict(item["params"])
            else:
                params = {k: v for k, v in item.items() if k not in {"id", "workflow"}}
            jobs.append(
                Job(id=job_id, params=params, workflow=item.get("workflow"))
            )

        out_dir_raw = data.get("output_dir") or "./output"
        return cls(
            workflow=data.get("workflow"),
            output_dir=Path(out_dir_raw).expanduser(),
            defaults=data.get("defaults") or {},
            jobs=jobs,
            raw_config=data.get("config") or {},
        )


# ---------------------------------------------------------------------- #
# Progress tracking
# ---------------------------------------------------------------------- #
class _Progress:
    def __init__(self, path: Path):
        self.path = path
        self.completed: set[str] = set()
        self.results: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                self.completed = set(data.get("completed") or [])
                self.results = data.get("results") or []
            except (json.JSONDecodeError, OSError):
                log.warning("Could not load progress file %s; starting fresh", self.path)

    def record(self, result: JobResult) -> None:
        if result.success:
            self.completed.add(result.job_id)
        self.results.append(asdict(result))
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"completed": sorted(self.completed), "results": self.results},
            indent=2,
        ))
        tmp.replace(self.path)


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #
class JobRunner:
    """Run a single workflow or a batch of jobs against a ComfyUI server."""

    def __init__(
        self,
        config: Config,
        client: Client | None = None,
        server: Server | None = None,
        registry: WorkflowRegistry | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._server = server
        self._owns_server = False
        self.registry = registry or default_registry

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.config.comfyui_host, port=self.config.comfyui_port)
        return self._client

    def ensure_server(self) -> None:
        """If a managed server is configured but not running, start it."""
        if self._server is not None:
            if not self._server.is_running():
                self._server.start()
            return
        if self.config.comfyui_path is None:
            return  # user is responsible for running their own server
        self._server = Server(
            comfyui_path=self.config.comfyui_path,
            host=self.config.comfyui_host,
            port=self.config.comfyui_port,
            output_dir=self.config.output_dir,
            input_dir=self.config.input_dir,
            python_exe=self.config.python_exe,
            extra_args=self.config.server_extra_args,
            start_timeout=self.config.server_start_timeout,
        )
        self._server.start()
        self._owns_server = True

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._owns_server and self._server is not None:
            self._server.stop()
            self._server = None
            self._owns_server = False

    def __enter__(self) -> JobRunner:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ #
    # Single run
    # ------------------------------------------------------------------ #
    def run_one(
        self,
        workflow: Workflow | str,
        params: dict[str, Any],
        output_dir: Path | None = None,
        job_id: str = "single",
    ) -> JobResult:
        wf = workflow if isinstance(workflow, Workflow) else self.registry.get(workflow)
        out_dir = (output_dir or self.config.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        merged_params = {**self.config.defaults, **params}
        ctx = RunContext(
            client=self.client,
            config=self.config,
            params=merged_params,
            output_dir=out_dir,
            job_id=job_id,
            registry=self.registry,
        )
        log.info("Running workflow %r (job=%s)", wf.name, job_id)
        start = time.monotonic()
        try:
            self.client.wait_ready(timeout=self.config.server_start_timeout)
            result = wf.run(ctx)
        except Exception as exc:  # noqa: BLE001 — captured into JobResult
            log.error("Job %s failed: %s", job_id, exc)
            return JobResult(
                job_id=job_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - start,
                metadata={"traceback": traceback.format_exc()},
            )
        return JobResult(
            job_id=job_id,
            success=True,
            files=[str(p) for p in result.files],
            duration_s=time.monotonic() - start,
            metadata=result.metadata,
        )

    # ------------------------------------------------------------------ #
    # Batch run
    # ------------------------------------------------------------------ #
    def run_batch(
        self,
        jobs_file: Path | str,
        resume: bool = False,
        on_result: Any = None,
    ) -> list[JobResult]:
        """Run every job in ``jobs_file`` and return the per-job results.

        Args:
            jobs_file: Path to a JSON jobs file (see :class:`Batch`).
            resume: If True, skip jobs whose ID is already marked completed in
                ``<output_dir>/.progress.json``.
            on_result: Optional callable invoked as ``on_result(result, index, total)``
                after each job completes. Useful for progress UIs.
        """
        batch = Batch.load(Path(jobs_file))
        output_dir = batch.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge the batch's `config` block on top of the runtime config (jobs file
        # overrides anything defined in standard locations, but is itself overridden
        # by config.defaults that the caller already applied).
        merged_config = self.config.merged_with(batch.raw_config) if batch.raw_config else self.config

        progress_path = output_dir / ".progress.json"
        progress = _Progress(progress_path)

        results: list[JobResult] = []
        total = len(batch.jobs)
        for idx, job in enumerate(batch.jobs):
            if resume and job.id in progress.completed:
                log.info("[%d/%d] Skipping completed job %s", idx + 1, total, job.id)
                continue
            wf_name = job.workflow or batch.workflow
            if not wf_name:
                raise WorkflowError(
                    f"No workflow specified for job {job.id} (set top-level `workflow` in the jobs file or per-job)"
                )

            params = {**merged_config.defaults, **batch.defaults, **job.params}
            log.info("[%d/%d] Running job %s with workflow %s", idx + 1, total, job.id, wf_name)
            result = self.run_one(
                workflow=wf_name,
                params=params,
                output_dir=output_dir,
                job_id=job.id,
            )
            progress.record(result)
            results.append(result)
            if on_result is not None:
                try:
                    on_result(result, idx, total)
                except Exception as exc:  # noqa: BLE001
                    log.warning("on_result callback raised: %s", exc)

        return results


# ---------------------------------------------------------------------- #
# Convenience
# ---------------------------------------------------------------------- #
def load_batch(path: Path | str) -> Batch:
    """Convenience wrapper around :meth:`Batch.load`."""
    return Batch.load(Path(path))


__all__ = [
    "Batch",
    "Job",
    "JobResult",
    "JobRunner",
    "MaxComfyError",
    "load_batch",
]
