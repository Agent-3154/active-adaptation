"""Evaluate a local training run for agent/watch scripts (no W&B API)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from active_adaptation.utils.experiment_logging import (
    ALGO_DIAGNOSTICS_JSONL_FILENAME,
    RUN_STATUS_FILENAME,
    assess_health,
    load_run_status,
)

# Exit codes for shell watchers / agents.
EXIT_GATE_PASSED = 0
EXIT_NOT_READY = 1
EXIT_FAIL = 2
EXIT_ERROR = 3
EXIT_COMPLETE = 4


def _is_finite_scalar(value: Any) -> bool:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value))


@dataclass(frozen=True)
class CheckResult:
    code: int
    status: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _load_last_algo_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_line = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def _parse_iso8601(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def evaluate_run(
    run_dir: Path | str,
    *,
    gate_iter: int = 50,
    warmup_iters: int = 3,
    max_kl: float | None = 0.5,
    min_explained_var: float | None = 0.0,
    stuck_minutes: float | None = 10.0,
    pid: int | None = None,
) -> CheckResult:
    run_dir = Path(run_dir)
    status_path = run_dir / RUN_STATUS_FILENAME
    algo_path = run_dir / ALGO_DIAGNOSTICS_JSONL_FILENAME

    details: dict[str, Any] = {"run_dir": str(run_dir)}

    status: dict[str, Any] | None = None
    if status_path.is_file():
        status = load_run_status(status_path)
        details["run_status"] = status

    algo_record = _load_last_algo_record(algo_path)
    if algo_record is not None:
        details["algo_last"] = algo_record

    if status is None and algo_record is None:
        return CheckResult(
            EXIT_ERROR,
            "error",
            "missing run_status.yaml and algo_diagnostics.jsonl",
            details,
        )

    state = str(status.get("state", "running")) if status else "running"
    iter_idx = int(
        status.get("iter", algo_record.get("iter", -1)) if status else algo_record["iter"]
    )
    details["iter"] = iter_idx

    if state == "completed":
        return CheckResult(EXIT_COMPLETE, "complete", "training completed", details)
    if state == "failed":
        return CheckResult(EXIT_FAIL, "fail", "run_status.state=failed", details)

    metrics = {}
    if algo_record is not None:
        metrics = dict(algo_record.get("metrics", {}))
    elif status is not None:
        metrics = dict(status.get("metrics", {}))

    health, health_issues = assess_health(metrics)
    if status is not None and status.get("health") == "fail":
        health = "fail"
        health_issues = list(status.get("health_issues", health_issues))

    if health == "fail":
        return CheckResult(
            EXIT_FAIL,
            "fail",
            "; ".join(health_issues) or "health=fail",
            {**details, "health_issues": health_issues},
        )

    if max_kl is not None:
        kl = metrics.get("actor/approx_kl")
        if _is_finite_scalar(kl) and abs(float(kl)) > max_kl:
            return CheckResult(
                EXIT_FAIL,
                "fail",
                f"actor/approx_kl>{max_kl} (last={float(kl):.3g})",
                details,
            )

    if min_explained_var is not None and "critic/explained_var" in metrics:
        explained_var = metrics.get("critic/explained_var")
        if _is_finite_scalar(explained_var) and float(explained_var) < min_explained_var:
            return CheckResult(
                EXIT_FAIL,
                "fail",
                f"critic/explained_var<{min_explained_var} (last={float(explained_var):.3g})",
                details,
            )

    monitor_pid = pid
    if monitor_pid is None and status is not None:
        raw_pid = status.get("pid")
        if isinstance(raw_pid, int):
            monitor_pid = raw_pid
        elif isinstance(raw_pid, str) and raw_pid.isdigit():
            monitor_pid = int(raw_pid)

    alive = _pid_alive(monitor_pid)
    details["pid"] = monitor_pid
    details["pid_alive"] = alive
    if alive is False:
        return CheckResult(
            EXIT_FAIL,
            "fail",
            f"training pid {monitor_pid} is not running",
            details,
        )

    if stuck_minutes is not None and status is not None:
        updated_at = _parse_iso8601(str(status.get("updated_at", "")))
        if updated_at is not None:
            age_min = (datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0
            details["status_age_min"] = age_min
            if alive and age_min > stuck_minutes:
                return CheckResult(
                    EXIT_FAIL,
                    "fail",
                    f"run_status stale for {age_min:.1f}m (>{stuck_minutes}m)",
                    details,
                )

    if iter_idx < warmup_iters:
        return CheckResult(
            EXIT_NOT_READY,
            "warming_up",
            f"iter {iter_idx} < warmup_iters {warmup_iters}",
            details,
        )

    if iter_idx < gate_iter:
        return CheckResult(
            EXIT_NOT_READY,
            "below_gate",
            f"iter {iter_idx} < gate_iter {gate_iter}",
            details,
        )

    return CheckResult(
        EXIT_GATE_PASSED,
        "gate_passed",
        f"iter {iter_idx} >= gate_iter {gate_iter}",
        {**details, "health": health, "health_issues": health_issues},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-iter", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--max-kl", type=float, default=0.5)
    parser.add_argument(
        "--min-explained-var",
        type=float,
        default=0.0,
        help="Fail when critic/explained_var is below this (if logged).",
    )
    parser.add_argument("--stuck-minutes", type=float, default=10.0)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = evaluate_run(
        args.run_dir,
        gate_iter=args.gate_iter,
        warmup_iters=args.warmup_iters,
        max_kl=args.max_kl,
        min_explained_var=args.min_explained_var,
        stuck_minutes=args.stuck_minutes,
        pid=args.pid,
    )
    payload = result.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"[{result.status}] exit={result.code} {result.message}")

    sys.exit(result.code)


if __name__ == "__main__":
    main()
