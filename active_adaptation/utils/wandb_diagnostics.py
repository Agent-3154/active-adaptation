from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import wandb

from active_adaptation.project_loading.manifest import CACHE_DIR

WANDB_DIAGNOSTICS_CACHE_DIR = CACHE_DIR / "wandb-diagnostics"


DEFAULT_KEYS: tuple[str, ...] = (
    # SAC / SAC-BAC / SAC-BEE
    "critic/grad_norm",
    "critic/q_loss",
    "critic/q_value",
    "critic/q_max",
    "critic/q_std",
    "critic/q_upper",
    "critic/q_lower",
    "critic/q_value_terminated",
    "critic/q_loss_terminated",
    "critic/prior_q_mean",
    "critic/prior_q_max",
    "critic/grad_norm",
    # PPO
    "critic/value_loss",
    "critic/explained_var",
    "actor/grad_norm",
    "actor/approx_kl",
    "actor/entropy",
    "actor/policy_loss",
    "actor/policy_gain",
    "actor/weighted_ratio",
    # Universal
    "critic/grad_norm",
)


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _to_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _extract_run_path(run_spec: str) -> tuple[str, str]:
    """
    Return (entity, project_and_run_path).

    Supports:
    - `run:<entity>/<project>/<run_id>[:<iter>]`
    - `https://wandb.ai/<entity>/<project>/runs/<run_id>`
    - `<entity>/<project>/<run_id>`
    """
    spec = run_spec.strip()
    if spec.startswith("run:"):
        spec = spec[len("run:") :]
        spec = spec.split(":", 1)[0]  # drop optional :<iter>

    if spec.startswith("http://") or spec.startswith("https://"):
        # Example: https://wandb.ai/ent/proj/runs/<id>
        m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", spec)
        if not m:
            raise ValueError(f"Unrecognized wandb URL: {run_spec!r}")
        entity, project, run_id = m.group(1), m.group(2), m.group(3)
        return entity, f"{project}/{run_id}"

    parts = spec.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Expected run spec of form `<entity>/<project>/<run_id>` or `run:<entity>/<project>/<run_id>`, got {run_spec!r}"
        )
    entity, project, run_id = parts[0], parts[1], "/".join(parts[2:])
    return entity, f"{project}/{run_id}"


def _extract_run_id(run_spec: str) -> str:
    _, rest = _extract_run_path(run_spec)
    return rest.split("/", 1)[1]


def default_cache_path(run_spec: str) -> Path:
    """Default JSON dump path for agent follow-up reads."""
    return WANDB_DIAGNOSTICS_CACHE_DIR / f"{_extract_run_id(run_spec)}.json"


def _resolve_run(run_spec: str) -> "wandb.wandb_sdk.wandb_run.Run":
    entity, rest = _extract_run_path(run_spec)
    # wandb.Api().run expects `entity/project/run_id` (or `entity/project/run_id:iter`)
    # We pass `entity/rest` where rest is `project/run_id`.
    run_path = f"{entity}/{rest}"
    api = wandb.Api()
    return api.run(run_path)


def _series_summary(values: list[tuple[int | None, float]]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "finite_count": 0,
            "nan_count": None,
            "min": None,
            "max": None,
            "last": None,
        }
    steps = [s for s, _ in values if s is not None]
    vs = [v for _, v in values]
    return {
        "count": len(values),
        "finite_count": len(values),
        "step_first": steps[0] if steps else None,
        "step_last": steps[-1] if steps else None,
        "min": min(vs),
        "max": max(vs),
        "last": vs[-1],
    }


def _detect_nonfinite(series: list[tuple[int | None, Any]]) -> dict[str, Any]:
    nonfinite_steps: list[int] = []
    for step, v in series:
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            nonfinite_steps.append(-1)
            continue
        if not math.isfinite(vf):
            if step is not None:
                nonfinite_steps.append(int(step))
            else:
                nonfinite_steps.append(-1)
    # Keep it small (agents don't need full indices).
    nonfinite_steps_sorted = sorted(set(nonfinite_steps))
    return {
        "nonfinite_count": len(nonfinite_steps),
        "nonfinite_steps_sample": nonfinite_steps_sorted[:20],
    }


@dataclass(frozen=True)
class DiagnosticsResult:
    run_id: str
    run_name: str
    run_url: str | None
    state: str | None
    metrics: dict[str, Any]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_url": self.run_url,
            "state": self.state,
            "metrics": self.metrics,
            "issues": self.issues,
        }


def analyze_run(
    run_spec: str,
    keys: Iterable[str] = DEFAULT_KEYS,
    *,
    samples: int = 3000,
) -> DiagnosticsResult:
    run = _resolve_run(run_spec)

    # `history()` returns a list of dicts when pandas=False.
    # Fetch each key separately because WandB sampled history for multiple keys
    # effectively intersects rows where all keys are present, which can be sparse
    # or empty for mixed metric families.
    key_list = list(keys)

    # History rows typically include `_step`.
    # Store raw series for small subset (for non-finite detection).
    metrics: dict[str, Any] = {}
    issues: list[str] = []

    # Pre-seed with empty; wandb can omit keys.
    raw_by_key: dict[str, list[tuple[int | None, Any]]] = {k: [] for k in key_list}

    for k in key_list:
        history = run.history(keys=[k], samples=samples, pandas=False)
        for row in history:
            step = row.get("_step")
            if k in row:
                raw_by_key[k].append((int(step) if step is not None else None, row.get(k)))

    # Summarize + heuristics.
    for k, raw in raw_by_key.items():
        finite_values: list[tuple[int | None, float]] = []
        for s, v in raw:
            vf = _to_float_or_none(v)
            if vf is None:
                continue
            finite_values.append((s, vf))

        metrics[k] = {
            "summary": _series_summary(finite_values),
            "nonfinite": _detect_nonfinite(raw),
        }

        nonfinite_count = metrics[k]["nonfinite"]["nonfinite_count"]
        if nonfinite_count and k in {"critic/grad_norm", "actor/grad_norm"}:
            issues.append(
                f"Non-finite values detected in `{k}` (count={nonfinite_count}). "
                "This often indicates AMP overflow or numerical instability in the critic loss."
            )

        if k == "critic/explained_var" and finite_values:
            # Explained variance should usually be in [−inf, 1], but consistently negative is a sign V tracking failed.
            last = finite_values[-1][1]
            if last < 0:
                issues.append(
                    "critic/explained_var is negative at the end of the run, "
                    "suggesting the value function is not fitting returns."
                )

        if k == "actor/approx_kl" and finite_values:
            last = finite_values[-1][1]
            if abs(last) > 0.5:
                issues.append(
                    f"actor/approx_kl is large at the end of the run (last={last:.3g}); "
                    "consider reducing learning rate or PPO clip / trust-region aggressiveness."
                )

    # Convenience issue for overall NaNs
    for grad_key in ("critic/grad_norm", "actor/grad_norm"):
        nonfinite = metrics.get(grad_key, {}).get("nonfinite", {}).get("nonfinite_count", 0)
        if nonfinite and nonfinite > 0 and grad_key == "critic/grad_norm":
            issues.append(
                "If NaNs appear intermittently, try: disable AMP, or switch scalar Q loss to Huber, "
                "or reduce bee_lambda / learning rate."
            )

    return DiagnosticsResult(
        run_id=str(getattr(run, "id", "")),
        run_name=str(getattr(run, "name", "")),
        run_url=getattr(run, "url", None),
        state=str(getattr(run, "state", "")) if hasattr(run, "state") else None,
        metrics=metrics,
        issues=issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize WandB diagnostics for algorithm analysis.")
    parser.add_argument(
        "--run",
        required=True,
        help="W&B run spec: run:<entity>/<project>/<run_id> or wandb.ai URL or <entity>/<project>/<run_id>",
    )
    parser.add_argument("--samples", type=int, default=3000, help="Max history points to fetch (most recent).")
    parser.add_argument("--keys", type=str, default="", help="Comma-separated metric keys. If empty, use DEFAULT_KEYS.")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Path to write JSON summary (default: .cache/wandb-diagnostics/<run_id>.json).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not write JSON to the default cache path.",
    )
    args = parser.parse_args()

    keys = DEFAULT_KEYS
    if args.keys.strip():
        keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())

    result = analyze_run(args.run, keys=keys, samples=args.samples)
    payload = result.to_dict()

    # Human readable first.
    if payload["issues"]:
        print("Issues:")
        for i, issue in enumerate(payload["issues"], 1):
            print(f"{i}. {issue}")
    else:
        print("Issues: none detected from the selected metric keys.")

    # Always print a compact JSON header so agents can parse.
    print("\nSummary JSON:")
    print(json.dumps(payload, indent=2))

    out_path: Path | None = None
    if args.out:
        out_path = Path(args.out)
    elif not args.no_cache:
        out_path = default_cache_path(args.run)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote cache: {out_path}")


if __name__ == "__main__":
    main()

