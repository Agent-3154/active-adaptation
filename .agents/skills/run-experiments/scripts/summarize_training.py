#!/usr/bin/env python3
"""Summarize local algo_diagnostics.jsonl and env_stats.jsonl for agent monitoring."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _median_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for record in records:
        metrics = record.get("metrics", record)
        value = metrics.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            values.append(float(value))
    if not values:
        return None
    return float(statistics.median(values))


def summarize_algo(path: Path, *, last: int) -> dict[str, Any]:
    records = _load_records(path)
    window = records[-last:] if last > 0 else records
    last_record = window[-1]
    metrics = last_record.get("metrics", {})
    return {
        "path": str(path),
        "iter_last": last_record.get("iter"),
        "env_frames_last": last_record.get("env_frames"),
        "rollout_fps_median": _median_metric(window, "performance/rollout_fps"),
        "rollout_fps_last": metrics.get("performance/rollout_fps"),
        "actor_approx_kl_last": metrics.get("actor/approx_kl"),
        "critic_explained_var_last": metrics.get("critic/explained_var"),
        "actor_grad_norm_last": metrics.get("actor/grad_norm"),
        "critic_grad_norm_last": metrics.get("critic/grad_norm"),
    }


def summarize_env(path: Path, *, last: int) -> dict[str, Any]:
    records = _load_records(path)
    window = records[-last:] if last > 0 else records
    last_record = window[-1]
    episode_keys = sorted(last_record.get("episode", {}).keys())
    ema_keys = sorted(last_record.get("ema", {}).keys())
    extra_keys = sorted(last_record.get("extra", {}).keys())
    return {
        "path": str(path),
        "iter_last": last_record.get("iter"),
        "env_frames_last": last_record.get("env_frames"),
        "episode_keys_last": episode_keys,
        "ema_last": last_record.get("ema", {}),
        "extra_last": last_record.get("extra", {}),
        "episode_nonempty_iters": sum(1 for r in window if r.get("episode")),
        "ema_keys": ema_keys,
        "extra_keys": extra_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="W&B run files dir")
    parser.add_argument("--last", type=int, default=10, help="Window for medians")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    algo_path = args.run_dir / "algo_diagnostics.jsonl"
    env_path = args.run_dir / "env_stats.jsonl"
    if not algo_path.is_file() or not env_path.is_file():
        print(f"Missing sidecar files under {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    result = {
        "algo": summarize_algo(algo_path, last=args.last),
        "env": summarize_env(env_path, last=args.last),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    algo = result["algo"]
    env = result["env"]
    print(f"run_dir: {args.run_dir}")
    print(
        f"algo iter={algo['iter_last']} rollout_fps="
        f"median={algo['rollout_fps_median']:.0f} last={algo['rollout_fps_last']:.0f}"
        if algo["rollout_fps_median"] is not None
        else f"algo iter={algo['iter_last']}"
    )
    print(
        f"  kl={algo['actor_approx_kl_last']} "
        f"explained_var={algo['critic_explained_var_last']} "
        f"actor_grad={algo['actor_grad_norm_last']} "
        f"critic_grad={algo['critic_grad_norm_last']}"
    )
    print(f"env iter={env['iter_last']} episode_iters={env['episode_nonempty_iters']}/{args.last}")
    if env["ema_last"]:
        print("  ema:")
        for key, value in sorted(env["ema_last"].items()):
            print(f"    {key}: {value}")
    if env["extra_last"]:
        print("  extra:")
        for key, value in sorted(env["extra_last"].items()):
            print(f"    {key}: {value}")
    if env["episode_keys_last"]:
        print(f"  episode keys (last nonempty): {env['episode_keys_last'][:8]}")


if __name__ == "__main__":
    main()
