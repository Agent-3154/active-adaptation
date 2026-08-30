#!/usr/bin/env python3
"""Summarize memory.jsonl for agent/human experiment monitoring."""

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
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _median_peak(records: list[dict[str, Any]], phase: str) -> float | None:
    values = [
        float(r["phases"][phase]["peak_allocated_MiB"])
        for r in records
        if r.get("phases", {}).get(phase, {}).get("peak_allocated_MiB") is not None
    ]
    if not values:
        return None
    return float(statistics.median(values))


def summarize(path: Path, *, last: int) -> dict[str, Any]:
    records = _load_records(path)
    if not records:
        raise ValueError(f"No records in {path}")
    window = records[-last:] if last > 0 else records
    latest = window[-1]
    train_op = latest.get("train_op") or []
    return {
        "path": str(path),
        "records_total": len(records),
        "window": len(window),
        "iter_first": window[0].get("iter"),
        "iter_last": latest.get("iter"),
        "num_envs": latest.get("num_envs"),
        "buffer_MiB": latest.get("buffer_MiB"),
        "peak_phase_last": latest.get("peak_phase"),
        "after_rollout_peak_allocated_MiB_median": _median_peak(window, "after_rollout"),
        "after_training_peak_allocated_MiB_median": _median_peak(window, "after_training"),
        "cuda_last": latest.get("cuda"),
        "phases_last": latest.get("phases"),
        "train_op_last": train_op,
        "train_op_peak_top": sorted(
            train_op,
            key=lambda row: float(row.get("peak_allocated_MiB", 0.0)),
            reverse=True,
        )[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="memory.jsonl path")
    parser.add_argument(
        "--last",
        type=int,
        default=3,
        help="Use last N records for medians (default: 3)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    result = summarize(args.path, last=args.last)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"memory: {result['path']}")
    print(
        f"iters {result['iter_first']}..{result['iter_last']} "
        f"(window {result['window']}/{result['records_total']}) "
        f"num_envs={result['num_envs']} buffer_MiB={result['buffer_MiB']}"
    )
    print(f"peak_phase (last): {result['peak_phase_last']}")
    rollout_peak = result["after_rollout_peak_allocated_MiB_median"]
    training_peak = result["after_training_peak_allocated_MiB_median"]
    if rollout_peak is not None:
        print(
            f"peak_allocated_MiB median: after_rollout={rollout_peak:.0f} "
            f"after_training={training_peak:.0f}"
        )
    cuda = result.get("cuda_last") or {}
    if cuda:
        print(
            "cuda (last after_training): "
            f"allocated={cuda.get('allocated_MiB', 0):.0f} "
            f"reserved={cuda.get('reserved_MiB', 0):.0f} "
            f"peak_allocated={cuda.get('peak_allocated_MiB', 0):.0f}"
        )
    print("train_op scopes (last iter, by peak_allocated):")
    for row in result["train_op_peak_top"]:
        print(
            f"  {row.get('peak_allocated_MiB', 0):6.0f} MiB peak  "
            f"{row.get('delta_allocated_MiB', 0):+6.0f} MiB delta  {row.get('path')}"
        )


if __name__ == "__main__":
    main()
