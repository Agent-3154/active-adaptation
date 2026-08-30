#!/usr/bin/env python3
"""Summarize profiling.jsonl for agent/human experiment monitoring."""

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


def _median_fps(records: list[dict[str, Any]]) -> float | None:
    values = [
        float(r["performance"]["rollout_fps"])
        for r in records
        if r.get("performance", {}).get("rollout_fps") is not None
    ]
    if not values:
        return None
    return float(statistics.median(values))


def _aggregate_bottlenecks(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_path: dict[str, list[float]] = {}
    for record in records:
        for timer in record.get("timers", []):
            path = str(timer.get("path", ""))
            pct = timer.get("pct")
            if not path or pct is None:
                continue
            by_path.setdefault(path, []).append(float(pct))
    ranked = sorted(
        (
            {
                "path": path,
                "pct_median": float(statistics.median(pcts)),
                "pct_max": max(pcts),
            }
            for path, pcts in by_path.items()
        ),
        key=lambda row: row["pct_median"],
        reverse=True,
    )
    return ranked[:top_k]


def summarize(path: Path, *, last: int, top_k: int) -> dict[str, Any]:
    records = _load_records(path)
    if not records:
        raise ValueError(f"No records in {path}")
    window = records[-last:] if last > 0 else records
    first_iter = window[0].get("iter")
    last_iter = window[-1].get("iter")
    return {
        "path": str(path),
        "records_total": len(records),
        "window": len(window),
        "iter_first": first_iter,
        "iter_last": last_iter,
        "backend": window[-1].get("backend"),
        "num_envs": window[-1].get("num_envs"),
        "rollout_fps_median": _median_fps(window),
        "rollout_fps_last": window[-1].get("performance", {}).get("rollout_fps"),
        "top_bottlenecks": _aggregate_bottlenecks(window, top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="profiling.jsonl path")
    parser.add_argument(
        "--last",
        type=int,
        default=1,
        help="Use last N records (default: 1 = latest iter only)",
    )
    parser.add_argument("--top", type=int, default=8, help="Top bottleneck paths to report")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    result = summarize(args.path, last=args.last, top_k=args.top)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"profiling: {result['path']}")
    print(
        f"iters {result['iter_first']}..{result['iter_last']} "
        f"(window {result['window']}/{result['records_total']}) "
        f"backend={result['backend']} num_envs={result['num_envs']}"
    )
    fps_med = result["rollout_fps_median"]
    fps_last = result["rollout_fps_last"]
    if fps_med is not None:
        print(f"rollout_fps: median={fps_med:.0f} last={fps_last:.0f}")
    print("top bottlenecks (median pct):")
    for row in result["top_bottlenecks"]:
        print(f"  {row['pct_median']:5.1f}%  {row['path']}")


if __name__ == "__main__":
    main()
