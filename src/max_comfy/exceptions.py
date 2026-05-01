"""Exception hierarchy for max-comfy-cli."""

from __future__ import annotations


class MaxComfyError(Exception):
    """Base class for all max-comfy errors."""


class WorkflowError(MaxComfyError):
    """A workflow definition is invalid or could not be loaded."""


class MissingParamError(WorkflowError):
    """A required template variable was not supplied."""

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        if self.available:
            super().__init__(
                f"Missing required parameter {name!r}. "
                f"Available: {', '.join(sorted(self.available))}"
            )
        else:
            super().__init__(f"Missing required parameter {name!r}.")


class ClientError(MaxComfyError):
    """The ComfyUI HTTP client encountered an error."""


class ComfyError(ClientError):
    """ComfyUI returned an error in workflow execution."""

    def __init__(self, message: str, prompt_id: str | None = None, details: dict | None = None):
        self.prompt_id = prompt_id
        self.details = details or {}
        super().__init__(message)


class ServerError(MaxComfyError):
    """A managed ComfyUI server subprocess could not be started or is unhealthy."""
