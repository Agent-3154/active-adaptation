---
name: run-experiments
description: Launch, monitor, and iterate on active-adaptation training experiments for agents and humans. Covers non-blocking runs, event-based watch_run.sh monitoring, check_run exit codes, run_status.yaml, JSONL sidecars (profiling, memory, algo, env), GPU memory scaling, smoke-vs-full gating, and early termination. Use when tuning hyperparameters, profiling rollout FPS or CUDA memory, running ablations, or orchestrating long train_ppo/train_offpolicy jobs.
---

# Run experiments (active-adaptation)

Agent-oriented playbook for **launch → monitor → decide → iterate**. Start here for experiment orchestration; use sibling skills for domain depth.

**Related skills**
- [simulation-performance](../simulation-performance/SKILL.md) — *how* to optimize hot paths (`ScopedTimer`, `torch.compile`, Isaac lazy-state caveats)
- [onpolicy-algorithms](../onpolicy-algorithms/SKILL.md) / [offpolicy-algorithms](../offpolicy-algorithms/SKILL.md) — algo config and training loop
- [wandb-diagnostics](../wandb-diagnostics/SKILL.md) — post-hoc W&B history when cloud/API is acceptable

**References:** [reference.md](reference.md) (schemas, exit codes, manifest template)

---

## Principles

1. **Decision value** — each run answers one hypothesis; stop when marginal info is low (see `AGENTS.md`).
2. **Smoke before full** — short `total_frames` + few envs before committing GPUs for hours.
3. **Non-blocking** — launch in background; never block the chat on a multi-hour job.
4. **Event-based monitoring** — local shell watches `run_status.yaml` / JSONL; wake the agent only on gate/fail/complete (token-efficient).
5. **Kill bad runs early** — startup errors, NaNs, stale heartbeat, or failed smoke gate.

---

## Local monitoring files (`wandb_run.dir/files/`)

| File | Contents | Update rate |
|------|----------|-------------|
| `run_status.yaml` | `state`, `iter`, `health`, key metrics, `memory`, `pid`, `updated_at` | Every log iter |
| `algo_diagnostics.jsonl` | `actor/*`, `critic/*`, `performance/*` | Every log iter |
| `env_stats.jsonl` | `episode` (`train/*`), `ema` (`reward.*`), `extra` | Every log iter |
| `profiling.jsonl` | Timer breakdown (PPO only) | Every PPO iter |
| `memory.jsonl` | CUDA snapshots + `train_op` scopes (PPO only) | Every PPO iter |

**Env toggles**

| Variable | Default | Effect |
|----------|---------|--------|
| `AA_METRICS_EXPORT` | `1` | `algo_diagnostics.jsonl`, `env_stats.jsonl`, `run_status.yaml` |
| `AA_MEMORY_EXPORT` | `1` | `memory.jsonl` |
| `AA_PROFILE_PRINT` | `1` | Terminal timer table each iter (set `0` for smoke) |
| `AA_MEMORY_SYNC` | `0` | `cuda.synchronize()` inside memory scopes (slower, more accurate) |

Written by `train_ppo.py` and `train_offpolicy.py` — **no W&B API** required for live monitoring.

---

## Hybrid monitoring (preferred for autonomous tuning)

**Do not** `/loop 5m` and re-read logs each tick (token-expensive). Instead:

1. Launch training in background; record `run_dir`, `pid`, command in `manifest.json`.
2. Start **local watcher** (cheap, no agent):

```bash
bash active-adaptation/.agents/skills/run-experiments/scripts/watch_run.sh \
  --run-dir /path/to/run/files \
  --pid <train_pid> \
  --gate-iter 50 \
  --poll-sec 30
```

3. Arm monitored shell on `^AGENT_WAKE_` (loop skill pattern) **or** wait for user to paste watcher output.
4. On wake, run **one agent turn**: read JSON payload → promote/kill/launch next run → update `EXPERIMENT.md`.

Optional progress heartbeats without a decision: `--heartbeat-min 60` (still emits `AGENT_WAKE`, use for status-only turns).

### `check_run` exit codes

```bash
uv run python -m active_adaptation.utils.check_run \
  --run-dir /path/to/run/files --gate-iter 50 --json
```

| Code | Meaning | Agent action |
|------|---------|--------------|
| `0` | Gate passed, healthy | Smoke decision: promote to full run or iterate hypothesis |
| `1` | Warming up / below gate | Do nothing; watcher keeps polling |
| `2` | Fail (NaN, KL, stale, dead pid) | Kill run; fix config; relaunch smoke |
| `3` | Missing files / error | Investigate launch path |
| `4` | `run_status.state=completed` | Collect results; next experiment |

Kill thresholds (defaults): `max_kl=0.5`, `min_explained_var=0.0`, `stuck_minutes=10`, `warmup_iters=3`.

---

## Launch workflow

```
Task Progress:
- [ ] State hypothesis and success criteria in EXPERIMENT.md
- [ ] Smoke command (small total_frames, few num_envs)
- [ ] Check GPU: nvidia-smi, tmux ls, pgrep
- [ ] Launch non-blocking; record run_dir + pid in manifest.json
- [ ] Start watch_run.sh (or manual check_run after ~gate iters)
- [ ] On AGENT_WAKE: decide promote / kill / tweak
- [ ] Log verdict; queue next manifest entry
```

### Smoke template (PPO + mjlab)

```bash
cd active-adaptation
AA_PROFILE_PRINT=0 uv run --project venv/mjlab python scripts/train_ppo.py \
  task=A2/A2LocoFlat algo=ppo_symaug 'algo.in_keys=[policy]' backend=mjlab \
  task.num_envs=64 total_frames=65536 \
  wandb.mode=disabled eval=false \
  checkpoint_interval=0 upload_interval=0
```

`A2LocoFlat` (and similar) need `algo.in_keys=[policy]` — command obs live under the policy group.

### Scale-up memory smoke (before large `num_envs`)

After algo gate passes at 64 envs, run a short memory smoke at target `num_envs`:

```bash
AA_PROFILE_PRINT=0 uv run --project venv/mjlab python scripts/train_ppo.py \
  task=A2/A2LocoFlat algo=ppo_symaug 'algo.in_keys=[policy]' backend=mjlab \
  task.num_envs=2048 total_frames=327680 \
  wandb.mode=disabled eval=false checkpoint_interval=0 upload_interval=0
```

Then `summarize_memory.py --last 5`. Promote only if `peak_allocated` is comfortably below device capacity (see below).

### Remote / long runs

```bash
tmux new -s exp-<key> 'AA_PROFILE_PRINT=0 uv run --project venv/mjlab python scripts/train_ppo.py ... 2>&1 | tee records/<key>.log'
```

---

## Decision channels

| Question | Read this |
|----------|-----------|
| Sim bottlenecks | `profiling.jsonl` + `summarize_profiling.py` |
| GPU headroom / OOM risk | `memory.jsonl` + `summarize_memory.py` |
| Algo stability | `algo_diagnostics.jsonl` or `check_run --json` |
| Reward / curriculum | `env_stats.jsonl` (`ema`, sparse `episode`) |
| Stuck / alive | `run_status.yaml` (`updated_at`, `pid`) |
| Startup crash | `records/*.log` first 5 min |

Summarize helpers:

```bash
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_training.py --run-dir <run_dir> --last 10
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_profiling.py --path <run_dir>/profiling.jsonl --last 10
python active-adaptation/.agents/skills/run-experiments/scripts/summarize_memory.py --path <run_dir>/memory.jsonl --last 5
```

---

## Simulation performance tuning

1. Smoke → wait `warmup_iters` → `summarize_profiling.py --last 10`.
2. Top `timers[].path` under `rollout/` → one code change (see simulation-performance skill).
3. Re-smoke; compare median `rollout_fps`.
4. Promote only if warm FPS improves without physics drift.

---

## GPU memory profiling & scaling

`memory.jsonl` records per-iter CUDA counters at **`after_rollout`** and **`after_training`**, plus scoped deltas inside `ppo_symaug.train_op` only.

### How to read it

| Field | Meaning |
|-------|---------|
| `buffer_MiB` | Static rollout tensor estimate (not peak) |
| `phases.*.peak_allocated_MiB` | Iter peak since `reset_peak_memory_stats()` at iter start |
| `phases.*.reserved_MiB` | PyTorch allocator pool — **not** a leak by itself |
| `peak_phase` | Which phase hit highest `peak_allocated_MiB` |
| `train_op[]` | Scoped peaks inside training (`compute_advantage`, `ppo_epochs`, `post_update`) |

**Typical pattern (A2LocoFlat, mjlab, `ppo_symaug`, 2048 envs):**
- `buffer_MiB` ~80–90; rollout peak ~0.5 GiB; training peak ~1.2 GiB on a 12 GiB GPU.
- `peak_phase` is usually **`after_training`**; `train_op/ppo_epochs` dominates (symmetry `torch.cat` + backward).

### Scaling checklist

1. Memory smoke at target `num_envs` (5+ iters, skip iter 0 for warmup).
2. `summarize_memory.py --last 5` → check median `after_training.peak_allocated_MiB`.
3. **Promote** if peak < ~80% of device memory; **reduce envs** or batch size if near OOM.
4. If `peak_allocated` **grows every iter** at fixed `num_envs` → suspect a leak; fix root cause.

### What *not* to do

- **Do not** call `torch.cuda.empty_cache()` every iter — reserved memory is allocator reuse; clearing slows the next iter and rarely lowers peak (live tensors still hold memory).
- **Do not** confuse `reserved_MiB` with wasted memory when plenty of headroom remains.
- Reserve `empty_cache()` for OOM recovery, train→eval phase changes, or leak diagnosis — not the default training loop.

**If OOM at scale:** reduce `num_envs`, switch `StackingCollector` → `BufferCollector`, reduce sym-aug batch doubling, or lower `train_every` — before cache clearing.

---

## Algorithm tuning

1. Smoke with sidecars (`wandb.mode=disabled` OK).
2. `check_run --gate-iter 50` → gate on stability (`actor/approx_kl`, `critic/explained_var`, grad norms).
3. Use `env_stats.jsonl` EMA for reward trends (episode stats lag `log_interval`).
4. Full run only after gate `0`. W&B optional for human dashboards.

---

## Autonomous tuning policy (user template)

Copy into the task when requesting unattended work:

```
- Smoke gate: iter >= 50
- Kill on: check_run exit 2, or pid exit non-zero
- Promote on: gate 0 + hypothesis-specific metric threshold
- Memory gate (optional): peak_allocated < 80% GPU after scale-up smoke
- Watcher: watch_run.sh --poll-sec 30 --gate-iter 50
- Heartbeat: none (or --heartbeat-min 60 for progress only)
```

Agent: launch job + watcher in background; **do not** poll via repeated chat turns unless watcher fires.

---

## Anti-patterns

- Fixed `/loop` polling every few minutes on multi-hour runs (token burn)
- Parsing terminal YAML instead of JSONL / `check_run`
- W&B API for live smoke monitoring when sidecars exist
- Algo tuning when `profiling.jsonl` shows >70% time in `rollout/`
- Deciding on iter 0–2 before warmup (CUDA/Warp JIT skews iter 0)
- Proactive `empty_cache()` in the training loop
- Scaling `num_envs` without a memory smoke when approaching GPU limits
