"""Experiment logger used by the train scripts.

W&B is the default. ``wandb.backend=swanlab`` switches the same run object
(``name``, ``id``, ``dir``, ``config``, ``summary``, ``log``, ``save``,
``finish``) to SwanLab. SwanLab is imported only when that backend is selected.

SwanLab's experiment name cannot be changed after ``init``. The train scripts
assign ``run.name`` before touching ``run.dir``, so this wrapper delays
``swanlab.init`` until the files directory, a log call, or ``save``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

BACKENDS = ("wandb", "swanlab")

_active: ExperimentRun | None = None


def init_run(cfg) -> ExperimentRun:
    """Open a run from a ``WandbConfig`` node. Default backend is W&B."""
    global _active
    run = ExperimentRun(cfg)
    _active = run
    return run


def Image(data: Any):
    """Image payload for ``run.log``. Accepts a matplotlib figure on both backends."""
    backend = _active.backend if _active is not None else "wandb"
    if backend == "swanlab":
        swanlab = _import_swanlab()
        return swanlab.Image(data)
    import wandb

    return wandb.Image(data)


def _import_swanlab():
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "swanlab is not installed. Install it with "
            "`uv pip install swanlab` or the `swanlab` extra."
        ) from exc
    return swanlab


def _json_ready(value: Any) -> Any:
    """Plain JSON-like tree for SwanLab config (Hydra nodes included)."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _ConfigView:
    """``run.config.update`` / ``run.config[key] = value`` for both backends."""

    def __init__(self, owner: ExperimentRun) -> None:
        self._owner = owner
        self._pending: dict[str, Any] = {}

    def update(self, data: Mapping[str, Any]) -> None:
        if self._owner.backend == "wandb":
            self._owner._raw.config.update(data)
            return
        if self._owner._raw is None:
            self._pending.update(dict(data))
            return
        self._owner._raw.config.update(_json_ready(dict(data)))

    def __setitem__(self, key: str, value: Any) -> None:
        if self._owner.backend == "wandb":
            self._owner._raw.config[key] = value
            return
        self.update({key: value})


class _SummaryView:
    """``run.summary[key] = value``. SwanLab stores these on the run config."""

    def __init__(self, owner: ExperimentRun) -> None:
        self._owner = owner

    def __setitem__(self, key: str, value: Any) -> None:
        raw = self._owner._ensure()
        if self._owner.backend == "wandb":
            raw.summary[key] = value
            return
        raw.config.update({key: _json_ready(value)})


class ExperimentRun:
    def __init__(self, cfg) -> None:
        backend = str(getattr(cfg, "backend", None) or "wandb")
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown experiment logger {backend!r}. Expected one of {BACKENDS}."
            )
        self.backend = backend
        self._cfg = cfg
        self._raw: Any = None
        self._id = uuid.uuid4().hex[:8]
        self._name = f"run-{self._id}"
        self.config = _ConfigView(self)
        self.summary = _SummaryView(self)
        if backend == "wandb":
            self._start_wandb()

    @property
    def id(self) -> str:
        if self._raw is not None:
            return str(self._raw.id)
        return self._id

    @property
    def name(self) -> str:
        if self.backend == "wandb" and self._raw is not None:
            return str(self._raw.name)
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)
        if self.backend == "wandb" and self._raw is not None:
            self._raw.name = self._name
            return
        if self.backend == "swanlab" and self._raw is not None:
            _log.warning(
                "SwanLab experiment name is fixed at init; keeping %s",
                getattr(self._raw, "name", self._name),
            )

    @property
    def dir(self) -> str:
        """Directory the train loop writes cfg, checkpoints, and JSONL into.

        W&B's ``run.dir`` is already that files directory. SwanLab's ``run.dir``
        is the run root, so this returns its ``files/`` subdirectory.
        """
        self._ensure()
        if self.backend == "wandb":
            return str(self._raw.dir)
        files = Path(self._raw.dir) / "files"
        files.mkdir(parents=True, exist_ok=True)
        return str(files)

    def log(self, data: dict) -> None:
        self._ensure()
        self._raw.log(data)

    def save(
        self,
        glob_str: str,
        policy: str = "now",
        base_path: str | None = None,
    ) -> None:
        self._ensure()
        if self.backend == "swanlab" and getattr(self._raw, "mode", None) == "disabled":
            return
        kwargs: dict[str, Any] = {"policy": policy}
        if base_path is not None:
            kwargs["base_path"] = base_path
        self._raw.save(glob_str, **kwargs)

    def finish(self) -> None:
        global _active
        try:
            if self.backend == "wandb":
                import wandb

                wandb.finish()
            elif self._raw is not None:
                self._raw.finish()
        finally:
            if _active is self:
                _active = None

    def _ensure(self) -> Any:
        if self._raw is None:
            self._start_swanlab()
        return self._raw

    def _start_wandb(self) -> None:
        import wandb

        self._raw = wandb.init(
            job_type=self._cfg.job_type,
            project=self._cfg.project,
            mode=self._cfg.mode,
            tags=self._cfg.tags,
        )

    def _start_swanlab(self) -> None:
        swanlab = _import_swanlab()
        tags = [str(tag) for tag in self._cfg.tags] if self._cfg.tags else []
        pending = _json_ready(self.config._pending) if self.config._pending else None
        experiment_name = self._name.replace("/", "-")
        self._raw = swanlab.init(
            project=str(self._cfg.project),
            experiment_name=experiment_name,
            job_type=str(self._cfg.job_type) if self._cfg.job_type else None,
            tags=tags,
            config=pending,
            mode=str(self._cfg.mode),
            id=self._id,
        )
        self.config._pending.clear()
        self._id = str(self._raw.id)
