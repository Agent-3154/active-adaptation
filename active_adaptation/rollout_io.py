"""Shared I/O helpers for rollout archives under ``scripts/rollout/``."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

ROLLOUT_FORMAT_VERSION = 1
ROLLOUT_FILENAME_RE = re.compile(r"^rollout_(\d+)_(\d+)(?:_[\w-]+)?$")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROLLOUT_ROOT = PACKAGE_DIR.parent / "scripts" / "rollout"


@dataclass
class RolloutInfo:
    task_algo: str
    timestamp: str
    pt_path: Path
    json_path: Path | None
    size_bytes: int
    policy_name: str
    num_steps: int
    num_envs: int
    num_keys: int

    @property
    def size_human(self) -> str:
        size = self.size_bytes
        if size >= 1024**3:
            return f"{size / (1024**3):.2f} GiB"
        if size >= 1024**2:
            return f"{size / (1024**2):.2f} MiB"
        if size >= 1024:
            return f"{size / 1024:.2f} KiB"
        return f"{size} B"

    @property
    def rel_path(self) -> str:
        try:
            return str(self.pt_path.relative_to(DEFAULT_ROLLOUT_ROOT))
        except ValueError:
            return str(self.pt_path)


def parse_rollout_filename(path: Path) -> tuple[int, int] | None:
    match = ROLLOUT_FILENAME_RE.match(path.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def key_to_str(key: str | tuple[str, ...]) -> str:
    if isinstance(key, str):
        return key
    return "/".join(key)


def key_from_str(key_str: str) -> str | tuple[str, ...]:
    if "/" not in key_str:
        return key_str
    return tuple(key_str.split("/"))


def describe_tensordict(tensordict: TensorDict) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key in tensordict.keys(include_nested=True):
        value = tensordict.get(key)
        if isinstance(value, torch.Tensor):
            shapes[key_to_str(key)] = list(value.shape)
    return shapes


def describe_tensor_entries(tensordict: TensorDict) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for key in tensordict.keys(include_nested=True):
        value = tensordict.get(key)
        if isinstance(value, torch.Tensor):
            entries[key_to_str(key)] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "size_bytes": int(value.numel() * value.element_size()),
            }
    return entries


def get_tensor_entries(metadata: dict, stacked: TensorDict) -> dict[str, dict[str, Any]]:
    entries = metadata.get("tensor_entries")
    if isinstance(entries, dict) and entries:
        return entries
    return describe_tensor_entries(stacked)


def episode_stats_to_metadata(stats: TensorDict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in stats.items(True, True):
        if isinstance(value, torch.Tensor):
            out[key_to_str(key)] = float(value.item())
    return dict(sorted(out.items()))


def write_metadata_json(metadata: dict, path: Path) -> None:
    def render(obj: Any, indent: int = 0) -> str:
        if isinstance(obj, list):
            return json.dumps(obj)
        if isinstance(obj, dict):
            pad = "  " * indent
            inner = "  " * (indent + 1)
            lines = ["{"]
            items = sorted(obj.items())
            for i, (key, value) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                lines.append(
                    f"{inner}{json.dumps(key)}: {render(value, indent + 1)}{comma}"
                )
            lines.append(f"{pad}}}")
            return "\n".join(lines)
        return json.dumps(obj)

    path.write_text(render(metadata) + "\n", encoding="utf-8")


def load_metadata(json_path: Path | None) -> dict:
    if json_path is None or not json_path.is_file():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_rollout(pt_path: Path, *, map_location: str | torch.device = "cpu") -> dict:
    payload: dict = torch.load(pt_path, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != ROLLOUT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported rollout format_version={version!r}; expected {ROLLOUT_FORMAT_VERSION}."
        )
    stacked = payload.get("stacked")
    if not isinstance(stacked, TensorDict) or len(stacked.batch_size) < 2:
        raise ValueError(
            f"Expected stacked TensorDict with batch [T, num_envs], got {type(stacked)}."
        )
    return payload


def save_rollout(pt_path: Path, payload: dict) -> None:
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(pt_path)


def save_rollout_with_metadata(
    pt_path: Path,
    payload: dict,
    metadata: dict,
) -> None:
    save_rollout(pt_path, payload)
    write_metadata_json(metadata, pt_path.with_suffix(".json"))


def list_tensor_keys(stacked: TensorDict) -> list[str]:
    return sorted(key_to_str(k) for k in stacked.keys(include_nested=True))


def update_metadata_shapes(metadata: dict, stacked: TensorDict) -> dict:
    metadata = dict(metadata)
    entries = describe_tensor_entries(stacked)
    metadata["tensor_entries"] = entries
    metadata["tensor_shapes"] = {key: info["shape"] for key, info in entries.items()}
    return metadata


def validate_rename_map(
    stacked: TensorDict, mapping: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    if not mapping:
        errors.append("Rename map is empty.")
        return mapping, errors

    existing = set(list_tensor_keys(stacked))
    cleaned: dict[str, str] = {}
    for old_key, new_key in mapping.items():
        old_key = old_key.strip()
        new_key = new_key.strip()
        if not old_key or not new_key:
            continue
        if old_key not in existing:
            errors.append(f"Key not found: {old_key!r}")
            continue
        cleaned[old_key] = new_key

    targets = list(cleaned.values())
    if len(targets) != len(set(targets)):
        errors.append("Rename map contains duplicate target keys.")

    final_keys = (existing - set(cleaned.keys())) | set(cleaned.values())
    if len(final_keys) != len(existing):
        errors.append("Rename map would collapse two keys into one.")

    return cleaned, errors


def apply_key_renames(stacked: TensorDict, mapping: dict[str, str]) -> TensorDict:
    cleaned, errors = validate_rename_map(stacked, mapping)
    if errors:
        raise ValueError("\n".join(errors))

    out = stacked.clone()
    for old_key, new_key in cleaned.items():
        out.rename_key_(key_from_str(old_key), key_from_str(new_key))
    return out


def validate_delete_keys(
    stacked: TensorDict, keys_to_delete: list[str]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if not keys_to_delete:
        errors.append("No keys selected for deletion.")
        return [], errors

    existing = set(list_tensor_keys(stacked))
    cleaned: list[str] = []
    for key in keys_to_delete:
        key = key.strip()
        if not key:
            continue
        if key not in existing:
            errors.append(f"Key not found: {key!r}")
            continue
        cleaned.append(key)

    if not cleaned:
        errors.append("No valid keys to delete.")
    elif len(cleaned) >= len(existing):
        errors.append("Cannot delete all keys.")

    return cleaned, errors


def apply_key_deletions(stacked: TensorDict, keys_to_delete: list[str]) -> TensorDict:
    cleaned, errors = validate_delete_keys(stacked, keys_to_delete)
    if errors:
        raise ValueError("\n".join(errors))
    td_keys = [key_from_str(key) for key in cleaned]
    return stacked.exclude(*td_keys)


def default_concat_key_name(source_keys: list[str]) -> str:
    return "_".join(source_keys)


def validate_key_concat(
    stacked: TensorDict,
    source_keys: list[str],
    dest_key: str,
    *,
    dim: int = -1,
    allow_overwrite: bool = False,
) -> tuple[list[str], str, list[str]]:
    errors: list[str] = []
    if len(source_keys) < 2:
        errors.append("Select at least two keys to concat.")
        return source_keys, dest_key, errors

    dest_key = dest_key.strip()
    if not dest_key:
        errors.append("Destination key name is empty.")
        return source_keys, dest_key, errors

    existing = set(list_tensor_keys(stacked))
    cleaned: list[str] = []
    for key in source_keys:
        key = key.strip()
        if not key:
            continue
        if key not in existing:
            errors.append(f"Key not found: {key!r}")
            continue
        cleaned.append(key)

    if len(cleaned) < 2:
        errors.append("Need at least two valid source keys.")
        return source_keys, dest_key, errors

    if dest_key in existing and dest_key not in cleaned and not allow_overwrite:
        errors.append(f"Destination key already exists: {dest_key!r}")

    tensors = [stacked.get(key_from_str(key)) for key in cleaned]
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        errors.append("All source keys must be tensors.")
        return cleaned, dest_key, errors

    ref = tensors[0]
    ndim = ref.ndim
    cat_dim = dim if dim >= 0 else ndim + dim
    if cat_dim < 0 or cat_dim >= ndim:
        errors.append(f"Invalid concat dim {dim} for rank-{ndim} tensors.")
        return cleaned, dest_key, errors

    ref_dtype = ref.dtype
    for key, tensor in zip(cleaned[1:], tensors[1:], strict=True):
        if tensor.dtype != ref_dtype:
            errors.append(
                f"dtype mismatch for {key!r}: {tensor.dtype} vs {ref_dtype}"
            )
        if len(tensor.shape) != ndim:
            errors.append(f"Rank mismatch for {key!r}.")
            continue
        for axis, (a, b) in enumerate(zip(ref.shape, tensor.shape, strict=True)):
            if axis == cat_dim:
                continue
            if a != b:
                errors.append(
                    f"Shape mismatch for {key!r} at dim {axis}: {a} vs {b}"
                )

    return cleaned, dest_key, errors


def apply_key_concat(
    stacked: TensorDict,
    source_keys: list[str],
    dest_key: str,
    *,
    dim: int = -1,
    allow_overwrite: bool = False,
) -> TensorDict:
    cleaned, dest, errors = validate_key_concat(
        stacked, source_keys, dest_key, dim=dim, allow_overwrite=allow_overwrite
    )
    if errors:
        raise ValueError("\n".join(errors))

    tensors = [stacked.get(key_from_str(key)) for key in cleaned]
    ndim = tensors[0].ndim
    cat_dim = dim if dim >= 0 else ndim + dim
    merged = torch.cat(tensors, dim=cat_dim)
    out = stacked.clone()
    out.set(key_from_str(dest), merged)
    return out


def resolve_out_path(
    path: Path,
    stacked: TensorDict,
    save_mode: str,
    suffix: str,
) -> Path:
    if save_mode == "In-place (backup)":
        backup_file(path)
        return path
    if save_mode == "In-place (no backup)":
        return path
    T, N = int(stacked.batch_size[0]), int(stacked.batch_size[1])
    tag = suffix.strip() or "edited"
    return path.with_name(f"rollout_{T}_{N}_{tag}.pt")


def validate_combine(paths: list[Path]) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if len(paths) < 2:
        errors.append("Select at least two rollouts to combine.")
        return [], errors

    entries: list[dict] = []
    for path in paths:
        try:
            payload = load_rollout(path)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        stacked: TensorDict = payload["stacked"]
        entries.append(
            {
                "path": path,
                "payload": payload,
                "stacked": stacked,
                "T": int(stacked.batch_size[0]),
                "N": int(stacked.batch_size[1]),
                "keys": set(list_tensor_keys(stacked)),
            }
        )

    if len(entries) < 2:
        return entries, errors

    ref = entries[0]
    ref_T = ref["T"]
    ref_keys = ref["keys"]
    for entry in entries[1:]:
        if entry["T"] != ref_T:
            errors.append(
                f"T mismatch: {ref['path'].name} has T={ref_T}, "
                f"{entry['path'].name} has T={entry['T']}"
            )
        if entry["keys"] != ref_keys:
            only_a = sorted(ref_keys - entry["keys"])
            only_b = sorted(entry["keys"] - ref_keys)
            if only_a:
                errors.append(f"Keys only in {ref['path'].name}: {only_a[:5]}")
            if only_b:
                errors.append(f"Keys only in {entry['path'].name}: {only_b[:5]}")

        for key in sorted(ref_keys & entry["keys"]):
            shape_a = ref["stacked"].get(key_from_str(key)).shape
            shape_b = entry["stacked"].get(key_from_str(key)).shape
            if len(shape_a) != len(shape_b):
                errors.append(f"Rank mismatch for {key!r}")
                continue
            for dim, (a, b) in enumerate(zip(shape_a, shape_b)):
                if dim == 1:
                    continue
                if a != b:
                    errors.append(
                        f"Shape mismatch for {key!r} at dim {dim}: {a} vs {b}"
                    )

    return entries, errors


def combine_rollouts(
    paths: list[Path],
    *,
    output_dir: Path | None = None,
) -> tuple[Path, dict]:
    entries, errors = validate_combine(paths)
    if errors:
        raise ValueError("\n".join(errors))

    stacked_parts = [entry["stacked"] for entry in entries]
    merged = torch.cat(stacked_parts, dim=1)
    T = int(merged.batch_size[0])
    N_sum = int(merged.batch_size[1])
    writer_max_size = max(int(entry["payload"].get("writer_max_size", T)) for entry in entries)

    payload = {
        "format_version": ROLLOUT_FORMAT_VERSION,
        "writer_max_size": writer_max_size,
        "stacked": merged,
    }

    policy_names = []
    for entry in entries:
        meta = load_metadata(entry["path"].with_suffix(".json"))
        name = meta.get("policy_name", "unknown")
        if name not in policy_names:
            policy_names.append(str(name))

    metadata = {
        "policy_name": ",".join(policy_names) if policy_names else "merged",
        "source_rollouts": [str(entry["path"]) for entry in entries],
        **update_metadata_shapes({}, merged),
    }

    out_dir = output_dir or entries[0]["path"].parent
    out_path = out_dir / f"rollout_{T}_{N_sum}_merged.pt"
    save_rollout_with_metadata(out_path, payload, metadata)
    return out_path, metadata


def discover_rollouts(root: Path = DEFAULT_ROLLOUT_ROOT) -> list[RolloutInfo]:
    if not root.is_dir():
        return []

    rollouts: list[RolloutInfo] = []
    for pt_path in sorted(root.rglob("rollout_*.pt")):
        if pt_path.suffix == ".tmp" or pt_path.name.endswith(".pt.tmp"):
            continue
        parsed = parse_rollout_filename(pt_path)
        if parsed is None:
            continue
        num_steps, num_envs = parsed

        rel_parts = pt_path.relative_to(root).parts
        task_algo = rel_parts[0] if len(rel_parts) >= 2 else pt_path.parent.name
        timestamp = rel_parts[1] if len(rel_parts) >= 2 else ""

        json_path = pt_path.with_suffix(".json")
        metadata = load_metadata(json_path if json_path.is_file() else None)
        tensor_shapes = metadata.get("tensor_shapes", {})

        rollouts.append(
            RolloutInfo(
                task_algo=task_algo,
                timestamp=timestamp,
                pt_path=pt_path,
                json_path=json_path if json_path.is_file() else None,
                size_bytes=pt_path.stat().st_size,
                policy_name=str(metadata.get("policy_name", "")),
                num_steps=num_steps,
                num_envs=num_envs,
                num_keys=len(tensor_shapes),
            )
        )
    return rollouts


def backup_file(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def delete_rollout_archive(pt_path: Path) -> None:
    pt_path = Path(pt_path)
    json_path = pt_path.with_suffix(".json")
    if pt_path.is_file():
        pt_path.unlink()
    if json_path.is_file():
        json_path.unlink()


def format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / (1024**3):.2f} GiB"
    if size >= 1024**2:
        return f"{size / (1024**2):.2f} MiB"
    if size >= 1024:
        return f"{size / 1024:.2f} KiB"
    return f"{size} B"
