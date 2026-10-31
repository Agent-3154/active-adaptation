"""Checkpoint resolve/download helpers (Comet remote + local files).

Remote specs::

    comet:<workspace>/<project>/<experiment_key>[:<iteration>]
    run:<workspace>/<project>/<experiment_key>[:<iteration>]   # alias

Local paths are unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import active_adaptation as aa

_log = logging.getLogger(__name__)

_CONFIG_FILE_NAMES = ("files/cfg.yaml", "cfg.yaml", "config.yaml")


def dict_flatten(a: dict, delim: str = ".") -> dict:
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update(
                {k + delim + kk: vv for kk, vv in dict_flatten(v, delim).items()}
            )
        else:
            result[k] = v
    return result


def _get_store_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    store_dir = repo_root / "scripts" / "comet"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def get_store_dir() -> Path:
    return _get_store_dir()


def _get_manifest_path() -> Path:
    return _get_store_dir() / "manifest.json"


def _load_manifest() -> dict[str, Any]:
    path = _get_manifest_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            _log.warning("Failed to read comet manifest.json, recreating.")
    return {"runs": {}}


def _save_manifest(data: dict[str, Any]) -> None:
    path = _get_manifest_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _lookup_manifest_run(run_path: str) -> dict[str, Any] | None:
    manifest = _load_manifest()
    runs = manifest.get("runs", {})
    for entry in runs.values():
        if entry.get("path") == run_path:
            return entry
    run_id = run_path.rstrip("/").split("/")[-1]
    return runs.get(run_id)


def _upsert_manifest(
    run_path: str,
    *,
    name: str,
    download_dir: Path,
    state: str | None = None,
) -> None:
    manifest = _load_manifest()
    runs = manifest.setdefault("runs", {})
    run_id = run_path.rstrip("/").split("/")[-1]
    runs[run_id] = {
        "id": run_id,
        "path": run_path,
        "name": name,
        "download_dir": str(download_dir),
        "state": state,
    }
    _save_manifest(manifest)


def _checkpoint_sort_key(name: str) -> float:
    iteration_str = Path(name).stem.split("_")[-1]
    if iteration_str == "final":
        return math.inf
    try:
        return float(int(iteration_str))
    except ValueError:
        return -1.0


def _find_local_cfg(download_dir: Path) -> Path | None:
    for candidate in (
        download_dir / "files" / "cfg.yaml",
        download_dir / "cfg.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _find_local_checkpoint(download_dir: Path, iteration: int | None) -> Path | None:
    if iteration is not None:
        path = download_dir / f"checkpoint_{iteration}.pt"
        return path if path.is_file() else None
    last = download_dir / "last_checkpoint.txt"
    if last.is_file():
        path = download_dir / last.read_text().strip()
        if path.is_file():
            return path
    checkpoints = sorted(
        (p for p in download_dir.glob("checkpoint_*.pt") if p.is_file()),
        key=lambda p: _checkpoint_sort_key(p.name),
    )
    return checkpoints[-1] if checkpoints else None


def parse_comet_run_spec(spec: str) -> tuple[str, int | None]:
    """Parse ``[comet:|run:]workspace/project/key[:iter]`` → (path, iteration)."""
    raw = spec.strip()
    for prefix in ("comet:", "run:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    iteration: int | None = None
    if ":" in raw.rsplit("/", 1)[-1]:
        head, _, maybe_iter = raw.rpartition(":")
        if maybe_iter.isdigit() and head.count("/") >= 1:
            raw = head
            iteration = int(maybe_iter)
    return raw, iteration


@dataclass(frozen=True)
class CachedRemoteRun:
    """Locally resolved remote-run assets (cfg + checkpoint)."""

    run_path: str
    run_id: str
    name: str
    download_dir: Path
    cfg_path: Path
    checkpoint_path: Path
    state: str | None = None


def try_get_cached_run(
    run_path: str, iteration: int | None = None
) -> CachedRemoteRun | None:
    run_path, parsed_iter = parse_comet_run_spec(run_path)
    if iteration is None:
        iteration = parsed_iter
    entry = _lookup_manifest_run(run_path)
    if entry is None:
        return None
    download_dir = Path(entry.get("download_dir") or "")
    if not download_dir.is_dir():
        name = entry.get("name")
        if name:
            download_dir = _get_store_dir() / str(name)
    if not download_dir.is_dir():
        return None
    cfg_path = _find_local_cfg(download_dir)
    checkpoint_path = _find_local_checkpoint(download_dir, iteration)
    if cfg_path is None or checkpoint_path is None:
        return None
    if iteration is not None or str(entry.get("state") or "").lower() in {
        "finished",
        "completed",
        "done",
    }:
        return CachedRemoteRun(
            run_path=entry.get("path") or run_path,
            run_id=str(entry.get("id") or run_path.rstrip("/").split("/")[-1]),
            name=str(entry.get("name") or download_dir.name),
            download_dir=download_dir,
            cfg_path=cfg_path,
            checkpoint_path=checkpoint_path,
            state=entry.get("state"),
        )
    return None


def _download_comet_run_assets(
    run_path: str,
    iteration: int | None = None,
) -> CachedRemoteRun:
    from comet_ml import API

    parts = run_path.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Comet run path must be workspace/project/experiment_key, got {run_path!r}"
        )
    workspace, project, exp_key = parts
    print(f"[CometCheckpoint] fetching {workspace}/{project}/{exp_key}")
    api = API()
    exp = api.get_experiment(workspace, project, exp_key)
    if exp is None:
        exp = api.get_experiment_by_key(exp_key)
    if exp is None:
        raise FileNotFoundError(f"Comet experiment not found: {run_path}")

    name = getattr(exp, "name", None) or exp_key
    root = _get_store_dir() / str(name).replace("/", "-")
    root.mkdir(parents=True, exist_ok=True)

    try:
        model_names = list(exp.get_model_names() or [])
    except Exception:
        model_names = []
    if "checkpoint" in model_names:
        model_dir = root / "models" / "checkpoint"
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CometCheckpoint] download_model checkpoint -> {model_dir}")
        exp.download_model("checkpoint", output_path=str(model_dir), expand=True)

    assets = exp.get_asset_list(asset_type="all") or []
    cfg_path: Path | None = None
    checkpoints: list[tuple[str, str, Path]] = []

    for asset in assets:
        file_name = str(asset.get("fileName") or "")
        asset_id = asset.get("assetId")
        if not asset_id:
            continue
        base = Path(file_name).name
        if base in {"cfg.yaml", "config.yaml"} or file_name.endswith("/cfg.yaml"):
            target = root / "cfg.yaml"
            if not target.is_file():
                data = exp.get_asset(asset_id, return_type="binary")
                target.write_bytes(data)
            cfg_path = target
            continue
        if "checkpoint" in base and base.endswith(".pt"):
            target = root / base
            if not target.is_file():
                data = exp.get_asset(asset_id, return_type="binary")
                target.write_bytes(data)
                print(f"[CometCheckpoint] wrote {target} ({target.stat().st_size} B)")
            checkpoints.append((base, str(asset_id), target))

    for pt in root.rglob("checkpoint_*.pt"):
        if pt.is_file() and not any(p == pt for *_, p in checkpoints):
            checkpoints.append((pt.name, "", pt))

    if cfg_path is None:
        cfg_path = _find_local_cfg(root)
    if cfg_path is None:
        raise FileNotFoundError(
            f"No cfg.yaml asset for Comet run {run_path}. "
            f"Expected one of {_CONFIG_FILE_NAMES} under {root}"
        )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files for Comet run {run_path}")

    if iteration is not None:
        chosen = next(
            (p for name, _, p in checkpoints if name == f"checkpoint_{iteration}.pt"),
            None,
        )
        if chosen is None:
            raise ValueError(f"Checkpoint {iteration} not found for {run_path}")
    else:
        chosen = sorted(
            (p for _, _, p in checkpoints),
            key=lambda p: _checkpoint_sort_key(p.name),
        )[-1]
        (root / "last_checkpoint.txt").write_text(chosen.name)

    state = getattr(exp, "get_state", lambda: None)()
    _upsert_manifest(
        f"{workspace}/{project}/{exp_key}",
        name=str(name),
        download_dir=root,
        state=str(state) if state is not None else "finished",
    )
    return CachedRemoteRun(
        run_path=f"{workspace}/{project}/{exp_key}",
        run_id=exp_key,
        name=str(name),
        download_dir=root,
        cfg_path=cfg_path,
        checkpoint_path=chosen,
        state=str(state) if state is not None else None,
    )


def resolve_comet_run(
    run_path: str,
    iteration: int | None = None,
    *,
    force: bool = False,
) -> CachedRemoteRun:
    """Resolve cfg + checkpoint for a Comet experiment."""
    run_path, parsed_iter = parse_comet_run_spec(run_path)
    if iteration is None:
        iteration = parsed_iter
    if not force:
        cached = try_get_cached_run(run_path, iteration)
        if cached is not None:
            print(
                f"[CometCheckpoint] using local cache for {cached.run_path} "
                f"(checkpoint={cached.checkpoint_path.name})"
            )
            return cached
    return _download_comet_run_assets(run_path, iteration)


class CheckpointBase:
    remote: bool = False

    def update(self) -> None:
        raise NotImplementedError

    def get_path(self) -> str | None:
        raise NotImplementedError

    def start_background_refresh(self, interval_sec: float) -> None:
        pass

    def stop_background_refresh(self) -> None:
        pass


class FileCheckpoint(CheckpointBase):
    remote: bool = False

    def __init__(self, path: str) -> None:
        self._path = path

    def update(self) -> None:
        pass

    def get_path(self) -> str | None:
        return self._path


class CometCheckpoint(CheckpointBase):
    """Checkpoint from Comet: ``comet:ws/proj/key[:iter]`` (``run:`` alias)."""

    remote: bool = True
    _DIST_WAIT_TIMEOUT_SEC = 300.0
    _DIST_POLL_INTERVAL_SEC = 0.5

    def __init__(self, spec: str) -> None:
        print(f"[CometCheckpoint] __init__ spec={spec}")
        self._spec = spec
        self._run_path, self._iteration = parse_comet_run_spec(spec)
        self._path: str | None = None
        self._lock = threading.Lock()
        self._refresh_interval_sec: float | None = None
        self._refresh_stop = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _download(self) -> str:
        resolved = resolve_comet_run(self._run_path, self._iteration)
        return str(resolved.checkpoint_path)

    def _download_rank0(self) -> str:
        world_size = aa.get_world_size()
        if world_size <= 1:
            return self._download()

        rank = aa.get_local_rank()
        marker_name = (
            ".dist_checkpoint_path_"
            + self._run_path.replace("/", "__").replace(":", "_")
            + ".json"
        )
        marker_path = _get_store_dir() / marker_name
        if rank == 0:
            try:
                path = self._download()
                payload = {"ok": True, "path": str(path), "error": None}
            except Exception as exc:
                payload = {"ok": False, "path": None, "error": repr(exc)}
                marker_tmp = marker_path.with_suffix(".json.tmp")
                marker_tmp.write_text(json.dumps(payload))
                marker_tmp.replace(marker_path)
                raise
            marker_tmp = marker_path.with_suffix(".json.tmp")
            marker_tmp.write_text(json.dumps(payload))
            marker_tmp.replace(marker_path)
            return str(path)

        start = time.perf_counter()
        while True:
            if marker_path.exists():
                payload = json.loads(marker_path.read_text())
                if not payload.get("ok"):
                    raise RuntimeError(
                        f"Rank 0 failed to prepare Comet checkpoint {self._run_path}: "
                        f"{payload.get('error')}"
                    )
                path = payload.get("path")
                if not path:
                    raise RuntimeError(f"Checkpoint marker missing path: {marker_path}")
                return path
            if time.perf_counter() - start > self._DIST_WAIT_TIMEOUT_SEC:
                raise TimeoutError(
                    f"Timed out waiting for rank 0 checkpoint marker: {marker_path}"
                )
            time.sleep(self._DIST_POLL_INTERVAL_SEC)

    def update(self) -> None:
        if self._path is None:
            path = self._download_rank0()
            with self._lock:
                self._path = path

    def get_path(self) -> str | None:
        with self._lock:
            return self._path

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(timeout=self._refresh_interval_sec or 60):
            try:
                path = self._download_rank0()
                with self._lock:
                    self._path = path
            except Exception as e:
                logging.warning("Background checkpoint refresh failed: %s", e)

    def start_background_refresh(self, interval_sec: float) -> None:
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self._refresh_interval_sec = interval_sec
        self._refresh_stop.clear()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def stop_background_refresh(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None


def parse_checkpoint(spec: str | None) -> CheckpointBase | None:
    """Build a checkpoint object from a local path or ``comet:`` / ``run:`` URI."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    spec = str(spec).strip()
    if spec.startswith(("comet:", "run:")):
        return CometCheckpoint(spec)
    return FileCheckpoint(spec)


def parse_checkpoint_path(path: str | None) -> str | None:
    checkpoint = parse_checkpoint(path)
    if checkpoint is None:
        return None
    checkpoint.update()
    return checkpoint.get_path()
