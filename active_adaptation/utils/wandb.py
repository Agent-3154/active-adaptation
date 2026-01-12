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
import logging
import os
import math
import json
from pathlib import Path
from typing import Dict, Any

import wandb
from omegaconf import OmegaConf
from typing import Union


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


def parse_checkpoint_path(path: str=None):
    """
    Parse a checkpoint path from local or wandb.
    If `path` is of the form `run:<wandb_run_id>[:<iterations>]`, it will be downloaded from wandb.

    Args:
        path (str or None): Path to a checkpoint. 

    Returns:
        str: Path to the checkpoint.
    """
    if path is None:
        return None

    if path.startswith("run:"):
        api = wandb.Api()
        try:
            run_path, iteration_str = path[4:].split(":")
            iterations = int(iteration_str)
            run = api.run(run_path)
        except:
            run = api.run(path[4:])
            iterations = None
        root = _get_store_dir() / run.name
        root.mkdir(parents=True, exist_ok=True)

        checkpoints = []
        for file in run.files():
            print(file.name)
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name in ("files/cfg.yaml", "cfg.yaml", "config.yaml"):
                file.download(str(root), replace=True)
                _manifest_add_file(run, file.name, root / Path(file.name).name, kind="config")

        if iterations is not None:
            checkpoint = None
            for file in checkpoints:
                if file.name == f"checkpoint_{iterations}.pt":
                    checkpoint = file
                    break
            if checkpoint is None:
                raise ValueError(f"Checkpoint {iterations} not found")
        else:
            def sort_by_time(file):
                iteration_str = file.name[:-3].split("_")[-1]
                if iteration_str == "final":
                    return math.inf
                else:
                    return int(iteration_str)
            checkpoints.sort(key=sort_by_time)
            checkpoint = checkpoints[-1]
        path = str(root / checkpoint.name)
        print(f"Downloading checkpoint to {path}")
        checkpoint.download(str(root), exist_ok=True)
        # Try to parse iteration from filename
        try:
            iteration_str = Path(checkpoint.name).stem.split("_")[-1]
            iteration_val: Union[int, str] = int(iteration_str) if iteration_str.isdigit() else iteration_str
        except Exception:
            iteration_val = None
        _manifest_add_file(run, checkpoint.name, Path(path), kind="checkpoint", iteration=iteration_val)
    return path

