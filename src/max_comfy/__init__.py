"""max-comfy-cli — generic CLI for ComfyUI workflows.

Public API:
    Workflow            Base class for workflow plugins.
    TemplateWorkflow    Workflow loaded from a JSON template with {{var}} placeholders.
    register            Decorator to register a Workflow subclass.
    registry            Global WorkflowRegistry instance.
    Client              ComfyUI HTTP client.
    Server              ComfyUI server lifecycle helper (optional).
    Config              Runtime configuration.
    Job                 A single batch job.
    JobRunner           Batch runner with resume support.
    RunContext          Context object passed to Workflow.run().
    RunResult           Return type of Workflow.run().
    OutputFile          A file produced by ComfyUI.
"""

# Built-in workflows (video-chain, video-postprocess, reactor-face-swap) register
# themselves via @register at module import time.
from . import builtins as _builtins  # noqa: F401, E402  — side-effect import
from . import media  # noqa: F401, E402
from .client import Client, OutputFile
from .config import Config
from .exceptions import (
    ClientError,
    ComfyError,
    MaxComfyError,
    MissingParamError,
    ServerError,
    WorkflowError,
)
from .runner import Job, JobResult, JobRunner
from .server import Server
from .workflows import (
    RunContext,
    RunResult,
    TemplateWorkflow,
    Workflow,
    WorkflowRegistry,
    register,
    registry,
    validate_graph,
)

__version__ = "0.2.0"

__all__ = [
    "Client",
    "ClientError",
    "ComfyError",
    "Config",
    "Job",
    "JobResult",
    "JobRunner",
    "MaxComfyError",
    "MissingParamError",
    "OutputFile",
    "RunContext",
    "RunResult",
    "Server",
    "ServerError",
    "TemplateWorkflow",
    "Workflow",
    "WorkflowError",
    "WorkflowRegistry",
    "__version__",
    "media",
    "register",
    "registry",
    "validate_graph",
]
