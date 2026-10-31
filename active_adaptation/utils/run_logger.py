"""Experiment logger used by the train scripts.

Comet is the default. ``logger.backend=trackio`` or ``logger.backend=swanlab``
switches the same run object (``name``, ``id``, ``dir``, ``config``,
``summary``, ``log``, ``save``, ``finish``). Optional backends are imported
only when selected.

SwanLab / Trackio / Comet register the experiment name at ``init``. Train
scripts assign ``run.name`` before touching ``run.dir``, so those backends
delay ``init`` until the files directory, a log call, or ``save``.

Install optional extras into the **backend** uv env you train with::

    uv sync --project venv/isaac51 --extra trackio   # or swanlab
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

BACKENDS = ("comet", "swanlab", "trackio")

_active: ExperimentRun | None = None


def init_run(cfg) -> ExperimentRun:
    """Open a run from a ``LoggerConfig`` / ``WandbConfig`` node. Default: Comet."""
    global _active
    run = ExperimentRun(cfg)
    _active = run
    return run


def Image(data: Any):
    """Image payload for ``run.log``. Accepts a matplotlib figure on all backends."""
    backend = _active.backend if _active is not None else "comet"
    if backend == "swanlab":
        return _import_swanlab().Image(data)
    if backend == "trackio":
        return _trackio_image(data)
    return _CometImage(data)


class _CometImage:
    """Marker holding a matplotlib figure / path / array for Comet ``log_image``."""

    __slots__ = ("data",)

    def __init__(self, data: Any) -> None:
        self.data = data


def _import_swanlab():
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError(
            "swanlab is not installed. Install with "
            "`uv sync --project venv/<backend> --extra swanlab`."
        ) from exc
    return swanlab


def _import_trackio():
    try:
        import trackio
    except ImportError as exc:
        raise ImportError(
            "trackio is not installed. Install with "
            "`uv sync --project venv/<backend> --extra trackio`."
        ) from exc
    return trackio


def _import_comet():
    try:
        import comet_ml
    except ImportError as exc:
        raise ImportError(
            "comet_ml is not installed. It is a core dependency of active-adaptation; "
            "re-sync your backend env (`uv sync --project venv/<backend>`)."
        ) from exc
    return comet_ml


def _trackio_image(data: Any):
    trackio = _import_trackio()
    if hasattr(data, "savefig"):
        from PIL import Image as PILImage

        buf = io.BytesIO()
        data.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return trackio.Image(PILImage.open(buf).convert("RGB"))
    return trackio.Image(data)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _ConfigView:
    def __init__(self, owner: ExperimentRun) -> None:
        self._owner = owner
        self._pending: dict[str, Any] = {}

    def update(self, data: Mapping[str, Any]) -> None:
        if self._owner._raw is None or getattr(self._owner._raw, "mode", None) == "disabled":
            self._pending.update(dict(data))
            return
        payload = _json_ready(dict(data))
        if self._owner.backend == "comet":
            self._owner._raw.log_parameters(payload)
            self._owner._comet_params.update(payload)
            return
        self._owner._raw.config.update(payload)

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})


class _SummaryView:
    def __init__(self, owner: ExperimentRun) -> None:
        self._owner = owner

    def __setitem__(self, key: str, value: Any) -> None:
        raw = self._owner._ensure()
        if self._owner.backend == "comet":
            raw.log_other(key, _json_ready(value))
            return
        raw.config.update({key: _json_ready(value)})


class _DisabledRun:
    def __init__(self, run_id: str, name: str) -> None:
        self.id = run_id
        self.name = name
        self.config: dict[str, Any] = {}
        self.mode = "disabled"

    def log(self, data: dict) -> None:
        return

    def finish(self) -> None:
        return

    def end(self) -> None:
        return

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return


class ExperimentRun:
    def __init__(self, cfg) -> None:
        backend = str(getattr(cfg, "backend", None) or "comet")
        if backend == "wandb":
            raise ValueError(
                "logger.backend='wandb' was removed. Use 'comet' (default), "
                "'trackio', or 'swanlab'."
            )
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown experiment logger {backend!r}. Expected one of {BACKENDS}."
            )
        self.backend = backend
        self._cfg = cfg
        self._raw: Any = None
        self._id = uuid.uuid4().hex[:8]
        self._name = f"run-{self._id}"
        self._files_dir: Path | None = None
        self._comet_params: dict[str, Any] = {}
        self.config = _ConfigView(self)
        self.summary = _SummaryView(self)
        mode = str(getattr(cfg, "mode", "online"))
        if mode == "disabled":
            self._start_disabled()

    @property
    def id(self) -> str:
        if self._raw is not None and self.backend == "comet":
            key = getattr(self._raw, "get_key", None)
            if callable(key):
                return str(key())
            return str(getattr(self._raw, "id", self._id))
        if self._raw is not None:
            return str(self._raw.id)
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)
        if self._raw is None:
            return
        if getattr(self._raw, "mode", None) == "disabled":
            self._raw.name = self._name
            return
        if self.backend == "comet":
            set_name = getattr(self._raw, "set_name", None)
            if callable(set_name):
                set_name(self._name.replace("/", "-"))
            return
        _log.warning(
            "%s experiment name is fixed at init; keeping %s",
            self.backend,
            getattr(self._raw, "name", self._name),
        )

    @property
    def dir(self) -> str:
        """Directory for cfg, checkpoints, and JSONL sidecars."""
        self._ensure()
        if self.backend in ("comet", "trackio"):
            assert self._files_dir is not None
            return str(self._files_dir)
        files = Path(self._raw.dir) / "files"
        files.mkdir(parents=True, exist_ok=True)
        return str(files)

    def log(self, data: dict) -> None:
        self._ensure()
        if getattr(self._raw, "mode", None) == "disabled":
            return
        if self.backend == "comet":
            self._log_comet(data)
            return
        self._raw.log(data)

    def save(
        self,
        glob_str: str,
        policy: str = "now",
        base_path: str | None = None,
    ) -> None:
        del policy, base_path  # wandb-compat kwargs
        self._ensure()
        if getattr(self._raw, "mode", None) == "disabled":
            return
        if self.backend == "trackio":
            self._save_trackio(glob_str)
            return
        if self.backend == "comet":
            self._save_comet(glob_str)
            return
        # swanlab
        self._raw.save(glob_str)

    def finish(self) -> None:
        global _active
        try:
            if self._raw is None or getattr(self._raw, "mode", None) == "disabled":
                return
            if self.backend == "comet":
                self._raw.end()
            elif self.backend == "trackio":
                _import_trackio().finish()
            else:
                self._raw.finish()
        finally:
            if _active is self:
                _active = None

    def _ensure(self) -> Any:
        if self._raw is None:
            if self.backend == "comet":
                self._start_comet()
            elif self.backend == "swanlab":
                self._start_swanlab()
            elif self.backend == "trackio":
                self._start_trackio()
            else:
                raise RuntimeError(f"Logger backend {self.backend!r} was not started.")
        return self._raw

    def _start_disabled(self) -> None:
        self._files_dir = self._make_local_files_dir(self.backend, self._id)
        self._raw = _DisabledRun(self._id, self._name)

    def _start_comet(self) -> None:
        mode = str(self._cfg.mode)
        if mode == "disabled":
            self._start_disabled()
            return

        comet_ml = _import_comet()
        from comet_ml import ExperimentConfig

        workspace = getattr(self._cfg, "workspace", None) or os.environ.get(
            "COMET_WORKSPACE"
        )
        online = mode == "online"
        tags = [str(t) for t in (self._cfg.tags or [])]
        kwargs: dict[str, Any] = {
            "project_name": str(self._cfg.project),
            "online": online,
            "experiment_config": ExperimentConfig(
                name=self._name.replace("/", "-"),
                tags=tags or None,
                log_env_details=True,
                log_env_gpu=True,
                log_env_host=True,
                log_env_cpu=True,
                # AA already exports GPU/CPU JSONL; keep Comet env snapshot light.
                auto_output_logging="simple",
            ),
        }
        if workspace:
            kwargs["workspace"] = str(workspace)
        api_key = os.environ.get("COMET_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        self._raw = comet_ml.start(**kwargs)
        if self.config._pending:
            self.config.update(self.config._pending)
            self.config._pending.clear()
        if self._cfg.job_type:
            self._raw.log_other("job_type", str(self._cfg.job_type))
        self._id = self.id
        self._files_dir = self._make_local_files_dir("comet", self._id)
        self._log_comet_run_ref()

    def _comet_run_ref(self) -> str | None:
        """Return ``workspace/project/key`` for copy-paste into ``checkpoint_path``."""
        raw = self._raw
        if raw is None or getattr(raw, "mode", None) == "disabled":
            return None
        key = str(raw.get_key() if callable(getattr(raw, "get_key", None)) else self._id)
        workspace = getattr(raw, "workspace", None) or getattr(self._cfg, "workspace", None)
        project = getattr(raw, "project_name", None) or getattr(self._cfg, "project", None)
        url = getattr(raw, "url", None)
        if isinstance(url, str) and "comet.com/" in url:
            # https://www.comet.com/<workspace>/<project>/<key>
            tail = url.split("comet.com/", 1)[1].strip("/").split("/")
            if len(tail) >= 3:
                workspace, project, key = tail[0], tail[1], tail[2]
        if not workspace:
            workspace = os.environ.get("COMET_WORKSPACE")
        if not workspace or not project:
            return None
        return f"{workspace}/{project}/{key}"

    def _log_comet_run_ref(self) -> None:
        ref = self._comet_run_ref()
        if ref is None:
            return
        # Others tab + stdout — easy copy for checkpoint_path=comet:… / run:…
        self._raw.log_other("run_path", ref)
        self._raw.log_other("checkpoint_path", f"comet:{ref}")
        print(f"[comet] run_path={ref}", flush=True)
        print(f"[comet] checkpoint_path=comet:{ref}", flush=True)

    def _start_swanlab(self) -> None:
        swanlab = _import_swanlab()
        tags = [str(tag) for tag in self._cfg.tags] if self._cfg.tags else []
        pending = _json_ready(self.config._pending) if self.config._pending else None
        self._raw = swanlab.init(
            project=str(self._cfg.project),
            experiment_name=self._name.replace("/", "-"),
            job_type=str(self._cfg.job_type) if self._cfg.job_type else None,
            tags=tags,
            config=pending,
            mode=str(self._cfg.mode),
            id=self._id,
        )
        self.config._pending.clear()
        self._id = str(self._raw.id)

    def _start_trackio(self) -> None:
        mode = str(self._cfg.mode)
        if mode == "disabled":
            self._start_disabled()
            return

        pending = _json_ready(self.config._pending) if self.config._pending else {}
        if self._cfg.job_type:
            pending.setdefault("job_type", str(self._cfg.job_type))
        if self._cfg.tags:
            pending.setdefault("tags", [str(tag) for tag in self._cfg.tags])

        kwargs: dict[str, Any] = {
            "project": str(self._cfg.project),
            "name": self._name.replace("/", "-"),
            "config": pending or None,
            "embed": False,
            "auto_log_gpu": False,
            "auto_log_cpu": False,
        }
        want_cloud = mode == "online"
        if want_cloud:
            space_id = getattr(self._cfg, "space_id", None) or os.environ.get(
                "TRACKIO_SPACE_ID"
            )
            if space_id:
                kwargs["space_id"] = str(space_id)
            bucket_id = getattr(self._cfg, "bucket_id", None)
            if bucket_id:
                kwargs["bucket_id"] = str(bucket_id)
            private = getattr(self._cfg, "private", None)
            if private is not None:
                kwargs["private"] = bool(private)
            if not space_id:
                _log.info(
                    "Trackio mode=online without logger.space_id / TRACKIO_SPACE_ID; "
                    "logging locally."
                )
                want_cloud = False

        hub_offline_prev = os.environ.get("HF_HUB_OFFLINE")
        whoami_restore: tuple[Any, Any] | None = None
        if not want_cloud:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            trackio = _import_trackio()
            if not want_cloud:
                whoami_restore = _install_offline_hf_whoami()
            self._raw = trackio.init(**kwargs)
        finally:
            if whoami_restore is not None:
                hub_mod, original = whoami_restore
                hub_mod.whoami = original
            if not want_cloud:
                if hub_offline_prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = hub_offline_prev
        self.config._pending.clear()
        self._id = str(self._raw.id)
        self._files_dir = self._make_local_files_dir("trackio", self._id)

    def _make_local_files_dir(self, backend: str, run_id: str) -> Path:
        if backend == "trackio":
            try:
                from trackio.utils import TRACKIO_DIR

                root = Path(TRACKIO_DIR)
            except Exception:
                root = Path(__file__).resolve().parents[2] / "scripts" / "trackio"
        else:
            root = Path(__file__).resolve().parents[2] / "scripts" / backend
        files = root / "runs" / str(run_id) / "files"
        files.mkdir(parents=True, exist_ok=True)
        return files

    def _log_comet(self, data: Mapping[str, Any]) -> None:
        metrics: dict[str, Any] = {}
        step = data.get("env_frames")
        if not isinstance(step, int):
            step = data.get("iter")
        if not isinstance(step, int):
            step = None
        for key, value in data.items():
            if isinstance(value, _CometImage):
                self._raw.log_image(value.data, name=str(key), step=step)
                continue
            # swanlab/wandb-style Image duck type
            if type(value).__name__ == "Image" and not _is_number(value):
                img = getattr(value, "image", None) or getattr(value, "data", value)
                self._raw.log_image(img, name=str(key), step=step)
                continue
            if _is_number(value):
                metrics[str(key)] = float(value)
        if metrics:
            self._raw.log_metrics(metrics, step=step)

    def _save_comet(self, glob_str: str) -> None:
        path = Path(glob_str)
        if not path.is_file():
            _log.warning("Comet save skipped (not a file): %s", glob_str)
            return
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt", ".onnx"}:
            self._raw.log_model(
                name="checkpoint",
                file_or_folder=str(path),
                metadata={"filename": path.name},
            )
        self._raw.log_asset(file_data=str(path), file_name=path.name)

    def _save_trackio(self, glob_str: str) -> None:
        path = Path(glob_str)
        if path.is_file():
            suffix = path.suffix.lower()
            artifact_type = (
                "model" if suffix in {".pt", ".pth", ".ckpt", ".onnx"} else "file"
            )
            self._raw.log_artifact(
                str(path),
                name=path.name.replace("/", "-"),
                type=artifact_type,
            )
            return
        _import_trackio().save(glob_str)


def _install_offline_hf_whoami() -> tuple[Any, Any]:
    import huggingface_hub

    original = huggingface_hub.whoami

    def _offline_whoami(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("HF Hub offline (active-adaptation Trackio local mode)")

    huggingface_hub.whoami = _offline_whoami
    return huggingface_hub, original
