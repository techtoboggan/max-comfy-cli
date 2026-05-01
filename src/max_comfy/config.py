"""Runtime configuration for max-comfy-cli.

Config is loaded from (in order; later overrides earlier):

1. Built-in defaults.
2. Optional TOML file at ``~/.config/max-comfy/config.toml``.
3. Optional TOML file at ``./max-comfy.toml`` in the working directory.
4. Explicit ``--config`` path passed on the command line.
5. Job-file ``config`` block (when running ``batch``).
6. CLI flags (``--port``, ``--output``, etc).

Field names mirror the TOML keys so the merge is straightforward.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib


@dataclass
class Config:
    # ComfyUI server connection
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    comfyui_path: Path | None = None  # required only when spawning the server
    python_exe: str = "python"
    server_extra_args: list[str] = field(default_factory=list)
    server_start_timeout: float = 60.0

    # I/O
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    input_dir: Path | None = None  # ComfyUI input dir (for staging files between phases)

    # Workflow discovery
    workflow_dirs: list[Path] = field(default_factory=list)
    workflow_modules: list[str] = field(default_factory=list)  # importable module names

    # Polling
    poll_interval: float = 1.0
    completion_timeout: float = 7200.0  # 2h, matches typical long video jobs

    # Defaults merged into every job's params unless overridden.
    defaults: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, explicit: Path | None = None) -> Config:
        """Load and merge config from standard locations + an optional explicit path."""
        cfg = cls()
        for path in cls._search_paths(explicit):
            if path and path.is_file():
                cfg = cfg.merged_with(cls._read_toml(path))
        return cfg

    @staticmethod
    def _search_paths(explicit: Path | None) -> list[Path | None]:
        home = Path.home() / ".config" / "max-comfy" / "config.toml"
        cwd = Path.cwd() / "max-comfy.toml"
        return [home, cwd, explicit]

    @staticmethod
    def _read_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            return tomllib.load(f)

    def merged_with(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with the given overrides applied."""
        if not overrides:
            return self
        data = self.to_dict()
        for key, value in overrides.items():
            if key == "defaults" and isinstance(value, dict):
                merged = dict(data.get("defaults") or {})
                merged.update(value)
                data["defaults"] = merged
            else:
                data[key] = value
        return self.__class__.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        kwargs: dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in valid:
                continue
            if key in {"comfyui_path", "input_dir"}:
                kwargs[key] = Path(value).expanduser() if value else None
            elif key == "output_dir":
                kwargs[key] = Path(value).expanduser()
            elif key == "workflow_dirs":
                kwargs[key] = [Path(p).expanduser() for p in value]
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                out[f.name] = str(value)
            elif isinstance(value, list):
                out[f.name] = [str(v) if isinstance(v, Path) else v for v in value]
            else:
                out[f.name] = value
        return out

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def resolved_workflow_dirs(self) -> list[Path]:
        """Default workflow search dirs, expanded and existence-filtered."""
        env_dir = os.environ.get("MAX_COMFY_WORKFLOWS")
        candidates: list[Path] = []
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.extend(self.workflow_dirs)
        candidates.append(Path.home() / ".config" / "max-comfy" / "workflows")
        candidates.append(Path.cwd() / "workflows")
        seen: set[Path] = set()
        out: list[Path] = []
        for p in candidates:
            try:
                resolved = p.expanduser().resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
        return out
