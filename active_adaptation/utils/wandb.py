# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import wandb
from omegaconf import OmegaConf
from typing import Union

import active_adaptation as aa

from torch import distributed as torch_dist

def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run


def _get_store_dir() -> Path:
    """Return the shared W&B store directory under active-adaptation/scripts/wandb."""
    # File layout: <repo_root>/active-adaptation/active_adaptation/utils/wandb.py
    # We want:     <repo_root>/active-adaptation/scripts/wandb
    repo_root = Path(__file__).resolve().parents[2]
    store_dir = repo_root / "scripts" / "wandb"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _get_manifest_path() -> Path:
    return _get_store_dir() / "manifest.json"


def _load_manifest() -> Dict[str, Any]:
    manifest_path = _get_manifest_path()
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            logging.warning("Failed to read W&B manifest.json, recreating.")
    return {"runs": {}}


def _save_manifest(data: Dict[str, Any]) -> None:
    manifest_path = _get_manifest_path()
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(manifest_path)


def _upsert_run_entry(manifest: Dict[str, Any], run) -> Dict[str, Any]:
    """Ensure a run entry exists in manifest and return it."""
    run_id = getattr(run, "id", None) or getattr(run, "name", None)
    entity = getattr(run, "entity", None) or ""
    project = getattr(run, "project", None) or ""
    name = getattr(run, "name", None) or ""
    path = f"{entity}/{project}/{run_id}" if entity and project and run_id else ""
    runs = manifest.setdefault("runs", {})
    entry = runs.get(run_id, {})
    entry.update({
        "id": run_id,
        "entity": entity,
        "project": project,
        "name": name,
        "path": path,
        "download_dir": str((_get_store_dir() / name).resolve()),
        "files": entry.get("files", []),
        "checkpoints": entry.get("checkpoints", []),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    })
    runs[run_id] = entry
    return entry


def _manifest_add_file(run, file_name: str, local_path: Path, kind: str, iteration: Union[int, str, None] = None) -> None:
    manifest = _load_manifest()
    entry = _upsert_run_entry(manifest, run)
    record: Dict[str, Any] = {
        "name": file_name,
        "local_path": str(local_path),
        "kind": kind,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if kind == "checkpoint":
        record["iteration"] = iteration
        entry.setdefault("checkpoints", []).append(record)
    else:
        entry.setdefault("files", []).append(record)
    entry["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    _save_manifest(manifest)


def get_store_dir() -> Path:
    """Public: return the shared store dir for downloaded W&B assets."""
    return _get_store_dir()


def is_checkpoint_url(path: str | None) -> bool:
    """Return True if path is an http(s) checkpoint server URL."""
    if not path:
        return False
    return path.startswith("http://") or path.startswith("https://")


# Timeout (seconds) for checkpoint URL download to avoid blocking indefinitely.
CHECKPOINT_URL_TIMEOUT = 120

# Filename under cache dir to store server Last-Modified so we skip re-download when unchanged.
_CHECKPOINT_VERSION_FILE = "version.txt"


def _head_checkpoint_url(url: str, timeout: float = CHECKPOINT_URL_TIMEOUT) -> tuple[int, str | None]:
    """HEAD request to checkpoint URL. Returns (status_code, Last-Modified or None)."""
    request = Request(url, headers={"User-Agent": "active-adaptation-checkpoint-client"}, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as resp:
            last_modified = resp.headers.get("Last-Modified")
            return (resp.status, last_modified)
    except HTTPError as e:
        return (e.code, None)


def download_checkpoint_url(url: str, timeout: float = CHECKPOINT_URL_TIMEOUT) -> str:
    """
    Download a checkpoint from a URL (e.g. checkpoint server) to the local cache.
    Skips download if server sends Last-Modified and it matches our cached version.

    URL shape: http(s)://host:port/download/<date>/<run_name> (trailing /latest.pt optional)
    """
    parsed = urlparse(url)
    # Path like /download/2025-02-15/12-30-45-Velocity-ppo (Path accepts "/" on all platforms)
    path_part = parsed.path.strip("/")
    cache_dir = _get_store_dir() / path_part
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "latest.pt"
    version_path = cache_dir / _CHECKPOINT_VERSION_FILE

    # HEAD first: if server has Last-Modified and we have same version cached, skip download
    try:
        status, server_last_modified = _head_checkpoint_url(url, timeout=timeout)
        if status != 200:
            if status == 404:
                raise FileNotFoundError(
                    f"Checkpoint not found at {url} (404). "
                    "Check that the run exists under the server root and has latest.pt."
                )
            raise RuntimeError(f"Checkpoint server returned {status} for {url}")
        if server_last_modified and local_path.exists() and version_path.exists():
            cached_version = version_path.read_text().strip()
            if cached_version == server_last_modified:
                logging.debug("Checkpoint unchanged (Last-Modified %s), using cache", server_last_modified)
                return str(local_path)
    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:
        logging.debug("HEAD failed, will try full download: %s", e)

    request = Request(url, headers={"User-Agent": "active-adaptation-checkpoint-client"})
    logging.info("Downloading checkpoint from %s to %s", url, local_path)
    try:
        with urlopen(request, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Checkpoint server returned {resp.status} for {url}")
            last_modified = resp.headers.get("Last-Modified")
            print("Last-Modified: %s", last_modified)
            b = resp.read()
            print("b: %s", b)
            local_path.write_bytes(b)
            if last_modified:
                version_path.write_text(last_modified)
    except HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"Checkpoint not found at {url} (404). "
                "Check that the run exists under the server root and has latest.pt."
            ) from e
        raise RuntimeError(f"Checkpoint server error {e.code} for {url}") from e

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logging.info("Downloaded checkpoint from %s to %s (%.2f MB)", url, local_path, size_mb)
    return str(local_path)


class CheckpointBase:
    """Abstract checkpoint: can be updated (e.g. re-download) and yields a local path for loading."""
    remote: bool = False

    def update(self) -> None:
        """Refresh from source (no-op for static files, re-download for URL/wandb as needed)."""
        raise NotImplementedError

    def get_path(self) -> str | None:
        """Current local path to the .pt file, or None if no checkpoint. Call update() first for remote sources."""
        raise NotImplementedError

    def start_background_refresh(self, interval_sec: float) -> None:
        """Start a background thread that periodically refreshes the checkpoint (no-op if not supported)."""
        pass

    def stop_background_refresh(self) -> None:
        """Stop the background refresh thread if running."""
        pass


class FileCheckpoint(CheckpointBase):
    """Local checkpoint file path; update() is a no-op."""
    remote: bool = False

    def __init__(self, path: str) -> None:
        self._path = path

    def update(self) -> None:
        pass

    def get_path(self) -> str | None:
        return self._path


class WandbCheckpoint(CheckpointBase):
    """Checkpoint from wandb run: run:<entity>/<project>/<run_id>[:<iteration>]. Fetched once in update()."""
    remote: bool = True

    def __init__(self, spec: str, api: "wandb.Api | None" = None) -> None:
        init_start = time.perf_counter()
        print(f"[WandbCheckpoint] __init__ start spec={spec}")
        self._spec = spec
        rest = spec[4:]
        try:
            self._run_path, iter_str = rest.split(":", 1)
            self._iteration: int | None = int(iter_str)
        except (ValueError, TypeError):
            self._run_path = rest
            self._iteration = None
        print(
            f"[WandbCheckpoint] parsed run_path={self._run_path}, "
            f"iteration={self._iteration}"
        )
        if api is None:
            print("[WandbCheckpoint] creating wandb.Api()")
            api_start = time.perf_counter()
            self._api = wandb.Api()
            print(
                f"[WandbCheckpoint] wandb.Api() ready "
                f"({time.perf_counter() - api_start:.2f}s)"
            )
        else:
            self._api = api
            print("[WandbCheckpoint] using injected wandb.Api")
        self._run: "wandb.wandb_sdk.wandb_run.Run | None" = None
        self._path: str | None = None
        self._lock = threading.Lock()
        self._refresh_interval_sec: float | None = None
        self._refresh_stop = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        print(f"[WandbCheckpoint] __init__ done ({time.perf_counter() - init_start:.2f}s)")

    @property
    def run(self) -> "wandb.wandb_sdk.wandb_run.Run":
        if self._run is None:
            print(f"[WandbCheckpoint] fetching run metadata: {self._run_path}")
            run_start = time.perf_counter()
            self._run = self._api.run(self._run_path)
            print(
                f"[WandbCheckpoint] run metadata ready "
                f"({time.perf_counter() - run_start:.2f}s), run_name={self._run.name}"
            )
        return self._run

    def _download(self) -> str:
        print(f"[WandbCheckpoint] _download start for {self._run_path}")
        run = self.run
        root = _get_store_dir() / run.name
        root.mkdir(parents=True, exist_ok=True)
        checkpoints = []
        print("[WandbCheckpoint] listing run files")
        files_start = time.perf_counter()
        run_files = list(run.files())
        print(
            f"[WandbCheckpoint] listed {len(run_files)} files "
            f"({time.perf_counter() - files_start:.2f}s)"
        )
        for file in run_files:
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name in ("files/cfg.yaml", "cfg.yaml", "config.yaml"):
                file.download(str(root), replace=True)
                _manifest_add_file(run, file.name, root / Path(file.name).name, kind="config")
        if self._iteration is not None:
            checkpoint_file = None
            for file in checkpoints:
                if file.name == f"checkpoint_{self._iteration}.pt":
                    checkpoint_file = file
                    break
            if checkpoint_file is None:
                raise ValueError(f"Checkpoint {self._iteration} not found")
        else:
            def sort_by_time(file):
                iteration_str = file.name[:-3].split("_")[-1]
                if iteration_str == "final":
                    return math.inf
                try:
                    return int(iteration_str)
                except ValueError:
                    return -1
            checkpoints.sort(key=sort_by_time)
            checkpoint_file = checkpoints[-1]
        path = str(root / checkpoint_file.name)
        last_ckpt_file = root / "last_checkpoint.txt"
        
        if last_ckpt_file.exists() and (root / checkpoint_file.name).exists():
            if last_ckpt_file.read_text().strip() == checkpoint_file.name:
                logging.debug("Wandb checkpoint unchanged (%s), using cache", checkpoint_file.name)
                print(f"[WandbCheckpoint] using cached checkpoint: {checkpoint_file.name}")
                return path
            
        print(f"[WandbCheckpoint] downloading checkpoint file: {checkpoint_file.name}")
        download_start = time.perf_counter()
        checkpoint_file.download(str(root), exist_ok=True)
        size_mb = checkpoint_file.size / (1024 * 1024)
        print(
            f"[WandbCheckpoint] downloaded checkpoint to {path} "
            f"(size: {size_mb:.2f} MB, {time.perf_counter() - download_start:.2f}s)"
        )
        last_ckpt_file.write_text(checkpoint_file.name)
        iteration_str = Path(checkpoint_file.name).stem.split("_")[-1]
        iteration_val: Union[int, str] = int(iteration_str) if iteration_str.isdigit() else iteration_str
        
        _manifest_add_file(run, checkpoint_file.name, Path(path), kind="checkpoint", iteration=iteration_val)
        print("[WandbCheckpoint] _download done")
        return path
    
    def _download_rank0(self) -> str:
        world_size = aa.get_world_size()
        rank = aa.get_local_rank()
        
        if world_size > 1:
            marker_name = (
                ".dist_checkpoint_path_"
                + self._run_path.replace("/", "__").replace(":", "_")
                + ".txt"
            )
            marker_path = _get_store_dir() / marker_name
            if rank == 0:
                print(f"[WandbCheckpoint] distributed update: rank 0 downloading for world_size={world_size}")
                path = self._download()
                marker_path.write_text(path)
            else:
                env_rank = os.getenv("LOCAL_RANK", "?")
                print(
                    "[WandbCheckpoint] distributed update: "
                    f"rank {rank} (LOCAL_RANK={env_rank}) waiting at barrier"
                )

            torch_dist.barrier()
            if rank != 0:
                if not marker_path.exists():
                    raise RuntimeError(f"Checkpoint marker missing after barrier: {marker_path}")
                path = marker_path.read_text().strip()
                print(f"[WandbCheckpoint] rank {rank} using checkpoint path from marker: {path}")
        else:
            path = self._download()
        return path
    
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
        logging.info("Started background checkpoint refresh every %.0f s", interval_sec)

    def stop_background_refresh(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None


class UrlCheckpoint(CheckpointBase):
    """Checkpoint from checkpoint server URL. update() re-downloads so callers get the latest."""
    remote: bool = True

    def __init__(self, url: str) -> None:
        self._url = url
        self._path: str | None = None
        self._lock = threading.Lock()
        self._refresh_interval_sec: float | None = None
        self._refresh_stop = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def update(self) -> None:
        path = download_checkpoint_url(self._url)
        with self._lock:
            self._path = path

    def get_path(self) -> str | None:
        with self._lock:
            return self._path

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(timeout=self._refresh_interval_sec or 60):
            try:
                path = download_checkpoint_url(self._url)
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
        logging.info("Started background checkpoint refresh every %.0f s", interval_sec)

    def stop_background_refresh(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None


def parse_checkpoint(spec: str | None) -> CheckpointBase | None:
    """
    Build a checkpoint object from a spec. Supports:

    1. Local path: path to a .pt file → FileCheckpoint.
    2. Wandb run: run:<entity>/<project>/<run_id>[:<iteration>] → WandbCheckpoint.
    3. Checkpoint server URL: http(s)://host:port/download/<date>/<run_name> → UrlCheckpoint.

    Returns None if spec is None. Call update() then get_path() to load.
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    spec = str(spec).strip()
    if is_checkpoint_url(spec):
        return UrlCheckpoint(spec)
    if spec.startswith("run:"):
        return WandbCheckpoint(spec)
    return FileCheckpoint(spec)


def parse_checkpoint_path(path: str | None) -> str | None:
    """
    Resolve a checkpoint spec to a local file path (one-time). For periodic refresh use parse_checkpoint() and update().
    """
    checkpoint = parse_checkpoint(path)
    if checkpoint is None:
        return None
    checkpoint.update()
    return checkpoint.get_path()
