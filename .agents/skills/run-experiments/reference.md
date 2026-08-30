# Run experiments — reference

## `profiling.jsonl` record schema

One JSON object per line, appended each training iteration by `train_ppo.py`.

```json
{
  "iter": 31,
  "env_frames": 65536,
  "backend": "mjlab",
  "num_envs": 64,
  "train_every": 32,
  "performance": {
    "rollout_fps": 4485.0,
    "rollout_time": 0.456,
    "training_time": 0.082,
    "iter_time": 0.541
  },
  "roots": ["rollout", "training"],
  "total_s": 0.538,
  "timers": [
    {
      "path": "rollout",
      "depth": 0,
      "count": 1,
      "total_s": 0.456,
      "avg_ms": 456.0,
      "pct": 84.8
    },
    {
      "path": "rollout/env._step/simulation",
      "depth": 2,
      "count": 32,
      "total_s": 0.12,
      "avg_ms": 3.75,
      "pct": 22.3
    }
  ]
}
```

- `timers[].pct` — share of `total_s` (sum of root timers), not parent-relative.
- `timers` sorted by `pct` descending (hottest first).
- Export uses full tree depth; terminal `print_summary(depth=3)` truncates.

**Source:** `active_adaptation/utils/profiling.py` (`collect_summary`, `append_profiling_jsonl`).

---

## `memory.jsonl` record schema

One JSON object per training iteration (when `AA_MEMORY_EXPORT=1`, default on). CUDA counters in MiB; empty when no GPU.

```json
{
  "iter": 4,
  "env_frames": 327680,
  "num_envs": 2048,
  "buffer_MiB": 84.25,
  "phases": {
    "after_rollout": {
      "allocated_MiB": 150.7,
      "reserved_MiB": 1562.0,
      "peak_allocated_MiB": 479.3,
      "peak_reserved_MiB": 1562.0
    },
    "after_training": {
      "allocated_MiB": 286.7,
      "reserved_MiB": 1562.0,
      "peak_allocated_MiB": 1238.1,
      "peak_reserved_MiB": 1562.0
    }
  },
  "peak_phase": "after_training",
  "cuda": {
    "allocated_MiB": 286.7,
    "reserved_MiB": 1562.0,
    "peak_allocated_MiB": 1238.1,
    "peak_reserved_MiB": 1562.0
  },
  "train_op": [
    {
      "path": "train_op/ppo_epochs",
      "count": 1,
      "delta_allocated_MiB": 131.6,
      "delta_reserved_MiB": 0.0,
      "peak_allocated_MiB": 623.2,
      "peak_reserved_MiB": 0.0
    },
    {
      "path": "train_op/compute_advantage",
      "count": 1,
      "delta_allocated_MiB": 273.0,
      "delta_reserved_MiB": 0.0,
      "peak_allocated_MiB": 135.6,
      "peak_reserved_MiB": 0.0
    },
    {
      "path": "train_op/post_update",
      "count": 1,
      "delta_allocated_MiB": 74.6,
      "delta_reserved_MiB": 0.0,
      "peak_allocated_MiB": 0.0,
      "peak_reserved_MiB": 0.0
    }
  ]
}
```

### Field reference

| Field | Meaning |
|-------|---------|
| `buffer_MiB` | Static nbytes estimate of rollout TensorDict (`fake.expand(num_envs, train_every)`) |
| `phases.after_rollout` | Snapshot after collector returns, before `train_op` |
| `phases.after_training` | Snapshot after `train_op` |
| `peak_phase` | Phase with highest `peak_allocated_MiB` this iter |
| `cuda` | Copy of `after_training` snapshot (convenience) |
| `train_op[].peak_allocated_MiB` | Peak growth during that scope (since iter peak reset) |
| `train_op[].delta_allocated_MiB` | Net allocated change across scope (can differ from peak) |

### Instrumentation scope

- **Tier 2 (iter phases):** `train_ppo.py` — reset peak at iter start; snapshot after rollout / after training.
- **Tier 3 (train_op scopes):** `ppo_symaug.train_op` only — `memory_scope()` around `compute_advantage`, `ppo_epochs`, `post_update`. Other algos get tier 2 only.

### Env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `AA_MEMORY_EXPORT` | `1` | Write `memory.jsonl` |
| `AA_MEMORY_SYNC` | `0` | `cuda.synchronize()` on scope enter/exit |

`run_status.yaml` includes a `memory` block (from latest `after_training`) while `state=running`.

**Source:** `active_adaptation/utils/memory_profiling.py`.

Summarize:

```bash
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_memory.py \
  --path /path/to/memory.jsonl --last 5
```

### Interpreting peaks (A2LocoFlat smoke, mjlab, 2048 envs, 12 GiB GPU)

| Metric | Typical value | Notes |
|--------|---------------|-------|
| `buffer_MiB` | ~84 | Small vs sim + training |
| `after_rollout.peak_allocated_MiB` | ~480 | Sim + policy forward during rollout |
| `after_training.peak_allocated_MiB` | ~1240 | Training dominates iter peak |
| `train_op/ppo_epochs.peak_allocated_MiB` | ~620 | Symmetry aug + PPO backward |
| `reserved_MiB` | ~1560 | Stable allocator pool — not a leak |

**Do not** use `empty_cache()` proactively when `peak_allocated` is well below device capacity.

---

## `algo_diagnostics.jsonl` record schema

One JSON object per training iteration (when `AA_METRICS_EXPORT=1`, default on).

```json
{
  "iter": 31,
  "env_frames": 65536,
  "metrics": {
    "env_frames": 65536,
    "performance/rollout_fps": 4485.0,
    "performance/rollout_time": 0.456,
    "performance/training_time": 0.082,
    "performance/iter_time": 0.541,
    "actor/approx_kl": 0.012,
    "actor/entropy": 1.23,
    "critic/explained_var": 0.45
  }
}
```

Keys: `actor/*`, `critic/*`, `performance/*`, `env_frames`.

**Source:** `active_adaptation/utils/experiment_logging.py`.

---

## `env_stats.jsonl` record schema

```json
{
  "iter": 31,
  "env_frames": 65536,
  "episode": {
    "train/stats/loco/linvel_exp": 1.2
  },
  "ema": {
    "reward.loco/linvel_exp": 0.85
  },
  "extra": {
    "curriculum/distance_traveled": 3.4
  }
}
```

- `episode` — sparse; populated when episode stats flush (`train/*` keys).
- `ema` — reward term EMA from `env.stats_ema` every iter.
- `extra` — `env.extra` (curriculum, custom command stats) every iter.

Disable export: `AA_METRICS_EXPORT=0`.

---

## `run_status.yaml` schema

Live heartbeat written each monitoring iteration.

```yaml
state: running   # running | completed | failed
iter: 31
env_frames: 65536
pid: 1234567
updated_at: "2026-08-31T01:40:12.345678+00:00"
health: ok       # ok | warn | fail
health_issues: []
backend: mjlab
num_envs: 2048
metrics:
  performance/rollout_fps: 75246.0
  actor/approx_kl: 0.012
  actor/grad_norm: 0.37
  critic/grad_norm: 0.04
  critic/explained_var: -0.46
memory:
  allocated_MiB: 286.7
  reserved_MiB: 1562.0
  peak_allocated_MiB: 1238.1
  peak_reserved_MiB: 1562.0
```

`memory` is present while `state=running` (from `after_training` snapshot). Final `state=completed` write may omit it.

**Source:** `active_adaptation/utils/experiment_logging.py` (`write_run_status`, `assess_health`).

---

## `check_run` exit codes

| Code | Status | Meaning |
|------|--------|---------|
| 0 | `gate_passed` | `iter >= gate_iter`, warmup done, no fail signals |
| 1 | `warming_up` / `below_gate` | Keep polling |
| 2 | `fail` | NaN grads, KL too high, stale `updated_at`, dead pid |
| 3 | `error` | Missing sidecar files |
| 4 | `complete` | `run_status.state=completed` |

```bash
uv run python -m active_adaptation.utils.check_run \
  --run-dir /path/to/run/files \
  --gate-iter 50 --warmup-iters 3 \
  --max-kl 0.5 --stuck-minutes 10 \
  --pid <train_pid> --json
```

---

## `watch_run.sh`

Local polling loop (no agent tokens). Emits `AGENT_WAKE_<purpose> <json>` on decision codes 0/2/4.

```bash
bash active-adaptation/.agents/skills/run-experiments/scripts/watch_run.sh \
  --run-dir /path/to/run/files \
  --pid <train_pid> \
  --gate-iter 50 \
  --poll-sec 30 \
  --heartbeat-min 0
```

Pair with loop skill monitored shell on `^AGENT_WAKE_`.

---

## `manifest.json` template

```json
{
  "hypothesis": "Scale A2LocoFlat to 2048 envs on mjlab",
  "created": "2026-08-31",
  "runs": [
    {
      "key": "mem-smoke-2048",
      "command": "uv run --project venv/mjlab python scripts/train_ppo.py task=A2/A2LocoFlat algo=ppo_symaug 'algo.in_keys=[policy]' backend=mjlab task.num_envs=2048 total_frames=327680 wandb.mode=disabled eval=false",
      "gpu": 0,
      "seed": 42,
      "status": "passed",
      "wandb_id": null,
      "run_dir": "/tmp/wandb/run-abc/files",
      "pid": 1234567,
      "gate_iter": 5,
      "watch_poll_sec": 30,
      "profiling_jsonl": "/tmp/wandb/run-abc/files/profiling.jsonl",
      "memory_jsonl": "/tmp/wandb/run-abc/files/memory.jsonl",
      "rollout_fps_median": 75246,
      "peak_allocated_MiB": 1238,
      "peak_phase": "after_training",
      "top_bottleneck": "rollout/env._step/simulation",
      "verdict": "headroom OK on 12GiB GPU",
      "notes": "train_op/ppo_epochs dominates peak"
    }
  ]
}
```

---

## Monitor commands

```bash
# Find recent sidecars
find active-adaptation /tmp/wandb -name 'memory.jsonl' -mmin -30 2>/dev/null
find active-adaptation /tmp/wandb -name profiling.jsonl -mmin -30 2>/dev/null

# Policy check (exit code = decision)
uv run python -m active_adaptation.utils.check_run \
  --run-dir /path/to/run/files --gate-iter 50 --json

# Event watcher (background; wakes agent on decision)
bash active-adaptation/.agents/skills/run-experiments/scripts/watch_run.sh \
  --run-dir /path/to/run/files --pid <pid> --gate-iter 50

# Summarize sidecars
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_profiling.py \
  --path /path/to/profiling.jsonl --last 10
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_memory.py \
  --path /path/to/memory.jsonl --last 5
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_training.py \
  --run-dir /path/to/run/files --last 10

# Raw peek
tail -1 /path/to/memory.jsonl | python3 -m json.tool | head -60
wc -l /path/to/algo_diagnostics.jsonl

# W&B algo diagnostics (optional, post-hoc)
uv run python -m active_adaptation.utils.wandb_diagnostics \
  --run <entity>/<project>/<run_id> --samples 500
```

---

## File map

| File | Role |
|------|------|
| `scripts/train_ppo.py` | Sidecars + `profiling.jsonl` + `memory.jsonl` + `run_status.yaml` |
| `scripts/train_offpolicy.py` | Sidecars + `run_status.yaml` |
| `active_adaptation/utils/experiment_logging.py` | JSONL + `run_status.yaml` writers |
| `active_adaptation/utils/check_run.py` | Policy evaluation + exit codes |
| `active_adaptation/utils/profiling.py` | `ScopedTimer`, `profiling.jsonl` |
| `active_adaptation/utils/memory_profiling.py` | `ScopedMemoryTimer`, `memory.jsonl`, `memory_scope()` |
| `active_adaptation/learning/ppo/ppo_symaug.py` | Tier-3 `train_op` memory scopes |
| `.agents/skills/run-experiments/scripts/watch_run.sh` | Local event watcher |
| `.agents/skills/run-experiments/scripts/summarize_memory.py` | Memory JSONL summarizer |
| `active_adaptation/utils/wandb_diagnostics.py` | Post-hoc W&B (optional) |
| `active_adaptation/pipeline_io.py` | `run_state.yaml` at run end (pipeline handoff) |
| `AGENTS.md` | GPU/tmux/rsync experiment ops |

---

## Typical timer paths (mjlab locomotion)

| Path | Usually means |
|------|----------------|
| `rollout/env._step/simulation` | Physics step + scene update |
| `rollout/env._step/simulation/scene.update` | mjlab scene tensor refresh |
| `rollout/env._step/update_callbacks` | MDP randomizations / robot hooks |
| `rollout/env._step/env.compute_observation` | Observation terms |
| `rollout/env._step/env.compute_reward` | Reward terms |
| `training/ppo_update` | Learner update (sim-bound if rollout pct low) |

Add nested `ScopedTimer("my_region")` under the hot parent before optimizing (see simulation-performance skill).

---

## Typical memory scopes (`ppo_symaug`)

| Scope | Usually means |
|-------|----------------|
| `train_op/ppo_epochs` | Symmetry `torch.cat` + PPO minibatch updates (largest peak) |
| `train_op/compute_advantage` | GAE + advantage normalization |
| `train_op/post_update` | Post-update actor forward on `tensordict.copy()` |

Add `memory_scope("train_op/my_region")` inside `train_op` when debugging training memory — not in the env step tree.
