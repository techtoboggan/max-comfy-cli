"""Workflow plugin system.

Two ways to define a workflow:

1. **JSON template** (no Python required). Export an API-format workflow from
   ComfyUI ("Save (API Format)") and replace any value you want to parameterise
   with ``{{var}}``. Place the file in a workflow directory and it's auto-discovered.

2. **Python class** (full control). Subclass :class:`Workflow`, set ``name``,
   and override :meth:`Workflow.build` (returns a single graph) or
   :meth:`Workflow.run` (full multi-step orchestration). Decorate with
   :func:`register` to add it to the global registry, or place the module in a
   workflow directory.

Workflows are looked up by name. Use ``max-comfy list`` to see what's available.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import templates
from .exceptions import WorkflowError

if TYPE_CHECKING:
    from .client import Client, OutputFile
    from .config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Run context / result types
# ---------------------------------------------------------------------- #
@dataclass
class RunContext:
    """Everything a Workflow.run() implementation needs.

    ``registry`` is set by :class:`~max_comfy.runner.JobRunner` so workflows can
    compose other workflows (e.g. a chain workflow invoking a per-segment
    workflow). It's typed as Any here because the dataclass needs a default and
    ``WorkflowRegistry`` is defined later in this module.
    """

    client: Client
    config: Config
    params: dict[str, Any]
    output_dir: Path
    job_id: str = ""
    registry: Any = None  # WorkflowRegistry — Any to avoid forward reference
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """The result of a single workflow run."""

    files: list[Path] = field(default_factory=list)
    outputs: list[OutputFile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Base class
# ---------------------------------------------------------------------- #
class Workflow:
    """Base class for workflow plugins.

    Subclasses should set ``name`` and override either :meth:`build` (simple
    single-submission workflows) or :meth:`run` (multi-step orchestration).
    """

    #: Unique identifier used at the CLI: ``max-comfy run <name>``.
    name: str = ""
    #: Short human-readable description shown in ``max-comfy list``.
    description: str = ""
    #: Default values merged into params before validation/build.
    defaults: dict[str, Any] = {}
    #: Names of params that must be provided (after defaults are applied).
    required: tuple[str, ...] = ()

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a ComfyUI prompt graph dict.

        Override this for simple, single-submission workflows. The default
        :meth:`run` implementation calls this, submits the graph, waits for
        completion, and downloads outputs.
        """
        raise NotImplementedError(f"{type(self).__name__} must override build() or run()")

    def run(self, ctx: RunContext) -> RunResult:
        """Execute the workflow against a running ComfyUI client.

        Override for multi-step orchestration (e.g. generate -> decode, multi-segment
        videos, interactive refinement). The default behaviour is:

        1. Merge defaults into params.
        2. Validate required params.
        3. Call ``build(params)``.
        4. Submit, wait, download to ``ctx.output_dir``.
        """
        merged = {**self.defaults, **ctx.params}
        self.validate_params(merged)
        graph = self.build(merged)
        result = ctx.client.submit_and_wait(
            graph,
            timeout=ctx.config.completion_timeout,
            poll_interval=ctx.config.poll_interval,
            download_dir=ctx.output_dir,
        )
        return RunResult(
            files=result.downloaded,
            outputs=result.outputs,
            metadata={"prompt_id": result.prompt_id},
        )

    def validate_params(self, params: dict[str, Any]) -> None:
        missing = [k for k in self.required if k not in params]
        if missing:
            raise WorkflowError(
                f"Workflow {self.name!r} is missing required params: {', '.join(sorted(missing))}"
            )

    def merge_defaults(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convenience helper: returns ``{**self.defaults, **params}``."""
        return {**self.defaults, **params}


# ---------------------------------------------------------------------- #
# JSON template workflow
# ---------------------------------------------------------------------- #
class TemplateWorkflow(Workflow):
    """A workflow defined by a JSON file with ``{{var}}`` placeholders.

    Companion ``.toml`` (sidecar with the same stem) may declare metadata:

    .. code-block:: toml

        description = "SDXL text-to-image"
        required = ["prompt"]

        [defaults]
        steps = 30
        cfg = 7.0
        seed = 42
    """

    def __init__(
        self,
        path: Path,
        name: str | None = None,
        description: str = "",
        defaults: dict[str, Any] | None = None,
        required: tuple[str, ...] = (),
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise WorkflowError(f"Template file not found: {self.path}")
        try:
            self._template = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"Invalid JSON in {self.path}: {exc}") from exc
        self.name = name or self.path.stem
        self.description = description
        self.defaults = defaults or {}
        self.required = tuple(required)

    def placeholders(self) -> set[str]:
        """Names of all ``{{var}}`` placeholders in the template."""
        return templates.find_placeholders(self._template)

    def build(self, params: dict[str, Any]) -> dict[str, Any]:
        return templates.render(self._template, params)


# ---------------------------------------------------------------------- #
# Registry + discovery
# ---------------------------------------------------------------------- #
class WorkflowRegistry:
    """Global registry of available workflows."""

    def __init__(self) -> None:
        self._workflows: dict[str, Workflow] = {}

    def register(self, workflow: Workflow) -> Workflow:
        if not workflow.name:
            raise WorkflowError(f"Workflow {workflow!r} must define a non-empty `name`")
        if workflow.name in self._workflows and self._workflows[workflow.name] is not workflow:
            log.warning("Overwriting existing workflow %r", workflow.name)
        self._workflows[workflow.name] = workflow
        return workflow

    def get(self, name: str) -> Workflow:
        try:
            return self._workflows[name]
        except KeyError:
            raise WorkflowError(
                f"Unknown workflow {name!r}. Available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._workflows)

    def items(self) -> list[tuple[str, Workflow]]:
        return sorted(self._workflows.items())

    def clear(self) -> None:
        self._workflows.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._workflows

    def __len__(self) -> int:
        return len(self._workflows)


registry = WorkflowRegistry()


def register(workflow_cls_or_instance: type[Workflow] | Workflow) -> Any:
    """Decorator / function to add a workflow to the global registry.

    Accepts either a class (instantiated with no args) or an already-constructed
    instance. Returns the input so it can be used as a decorator:

    .. code-block:: python

        @register
        class MyFlow(Workflow):
            name = "my-flow"
            def build(self, params):
                return {...}
    """
    if isinstance(workflow_cls_or_instance, Workflow):
        registry.register(workflow_cls_or_instance)
        return workflow_cls_or_instance
    if isinstance(workflow_cls_or_instance, type) and issubclass(
        workflow_cls_or_instance, Workflow
    ):
        instance = workflow_cls_or_instance()
        registry.register(instance)
        return workflow_cls_or_instance
    raise WorkflowError(f"@register expects a Workflow class or instance, got {workflow_cls_or_instance!r}")


# ---------------------------------------------------------------------- #
# Auto-discovery
# ---------------------------------------------------------------------- #
def discover(
    dirs: list[Path] | None = None,
    modules: list[str] | None = None,
    target: WorkflowRegistry | None = None,
) -> WorkflowRegistry:
    """Populate the registry from filesystem and importable modules.

    For each directory in ``dirs``:

    * Every ``*.json`` file becomes a :class:`TemplateWorkflow`.
    * Every ``*.py`` file is imported under a sandbox module name; any
      ``Workflow`` subclasses inside that call :func:`register` (or are
      registered via the ``@register`` decorator) are added.

    For each name in ``modules``: ``importlib.import_module(name)`` is called
    and the same applies — ``register`` calls inside take effect.
    """
    if target is None:
        target = registry
    if dirs:
        for d in dirs:
            _discover_dir(d, target)
    if modules:
        for m in modules:
            try:
                importlib.import_module(m)
            except ImportError as exc:
                log.warning("Could not import workflow module %r: %s", m, exc)
    return target


def _discover_dir(directory: Path, target: WorkflowRegistry) -> None:
    if not directory.is_dir():
        return
    log.debug("Discovering workflows in %s", directory)
    for path in sorted(directory.iterdir()):
        if path.suffix == ".json":
            try:
                wf = _load_json_template(path)
                target.register(wf)
            except WorkflowError as exc:
                log.warning("Skipping %s: %s", path, exc)
        elif path.suffix == ".py" and not path.name.startswith("_"):
            _load_python_module(path)


def _load_json_template(path: Path) -> TemplateWorkflow:
    sidecar = path.with_suffix(".toml")
    description = ""
    defaults: dict[str, Any] = {}
    required: tuple[str, ...] = ()
    if sidecar.is_file():
        meta = _read_sidecar(sidecar)
        description = str(meta.get("description", ""))
        if isinstance(meta.get("defaults"), dict):
            defaults = meta["defaults"]
        if isinstance(meta.get("required"), list):
            required = tuple(str(s) for s in meta["required"])
    return TemplateWorkflow(
        path=path,
        name=path.stem,
        description=description,
        defaults=defaults,
        required=required,
    )


def _read_sidecar(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_python_module(path: Path) -> None:
    """Import a Python file as a module so its @register calls run."""
    module_name = f"max_comfy._user_workflows.{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        log.warning("Could not load %s", path)
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — surface user errors clearly
        log.warning("Error loading workflow module %s: %s", path, exc)
