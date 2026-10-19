#!/usr/bin/env python3
"""Print a markdown FPS / memory table for one benchmark/runs/<stamp> directory."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _median_timer(records: list[dict], timer_path: str) -> float | None:
    values: list[float] = []
    for record in records:
        for timer in record.get("timers", []):
            if timer.get("path") == timer_path:
                values.append(float(timer["total_s"]))
    if not values:
        return None
    return float(statistics.median(values))


def _smi_peak(case_dir: Path) -> int | None:
    smi = case_dir.parent / f"{case_dir.name}.smi"
    if not smi.is_file():
        return None
    values = [int(token) for token in smi.read_text().split() if token.strip().isdigit()]
    if not values:
        return None
    return max(values)


def _oom(case_dir: Path) -> bool:
    log = case_dir.parent / f"{case_dir.name}.log"
    if not log.is_file():
        return False
    text = log.read_text(errors="replace")
    return "out of memory" in text.lower() or "Failed to allocate" in text


def _envs_from_log(case_dir: Path) -> str:
    log = case_dir.parent / f"{case_dir.name}.log"
    if not log.is_file():
        return case_dir.parent.name
    for line in log.read_text(errors="replace").splitlines():
        if "task.num_envs=" in line:
            return line.split("task.num_envs=", 1)[1].split()[0].strip("',]")
    return case_dir.parent.name


def _case_row(case_dir: Path) -> str | None:
    profiling = case_dir / "profiling.jsonl"
    memory = case_dir / "memory.jsonl"
    if not profiling.is_file():
        if _oom(case_dir):
            return f"| {_envs_from_log(case_dir)} | {case_dir.name} | OOM | | | | |"
        return None
    records = _load(profiling)[1:]
    if not records:
        return None
    envs = records[-1].get("num_envs", _envs_from_log(case_dir))
    fps = statistics.median(float(record["performance"]["rollout_fps"]) for record in records)
    sim = _median_timer(records, "rollout/env._step/simulation")
    torch_peak = None
    if memory.is_file():
        mem_records = _load(memory)[1:]
        peaks = [
            float(record["phases"]["after_training"]["peak_allocated_MiB"])
            for record in mem_records
            if record.get("phases", {}).get("after_training", {}).get("peak_allocated_MiB") is not None
        ]
        if peaks:
            torch_peak = statistics.median(peaks)
    gpu = _smi_peak(case_dir)
    sim_s = "" if sim is None else f"{sim:.3f}"
    torch_s = "" if torch_peak is None else f"{torch_peak:.0f}"
    gpu_s = "" if gpu is None else str(gpu)
    return f"| {envs} | {case_dir.name} | {fps:.0f} | {sim_s} | {torch_s} | {gpu_s} |"


def summarize(run_dir: Path) -> str:
    rows = []
    seen: set[Path] = set()
    for profiling in sorted(run_dir.rglob("profiling.jsonl")):
        seen.add(profiling.parent.resolve())
        row = _case_row(profiling.parent)
        if row:
            rows.append(row)
    for log in sorted(run_dir.rglob("*.log")):
        case_dir = log.parent / log.stem
        if case_dir.resolve() in seen:
            continue
        if _oom(case_dir):
            rows.append(f"| {_envs_from_log(case_dir)} | {case_dir.name} | OOM | | | | |")
    header = (
        "| Envs | Case | Rollout FPS | Sim (s) | Torch peak (MiB) | GPU (MiB) |\n"
        "|---:|---|---:|---:|---:|---:|"
    )
    machine = run_dir / "machine.txt"
    title = f"# {run_dir.name}"
    if machine.is_file():
        title += "\n\n```\n" + machine.read_text().strip() + "\n```"
    note = (
        "\n\nMedians of iterations 1–5. `Sim` is seconds inside the `simulation` timer "
        "for one rollout. That timer is the CUDA-graph launch. The first reward that "
        "reads body state waits for the graph, so a heavy contact step shows up there "
        "rather than in `Sim`. GPU is the max `nvidia-smi` sample at 1 Hz."
    )
    if not rows:
        return title + "\n\nNo finished cases.\n"
    return title + note + "\n\n" + header + "\n" + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.run_dir), end="")


if __name__ == "__main__":
    main()
