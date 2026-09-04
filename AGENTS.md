# Repository Guidelines

## Project Structure & Module Organization
`active_adaptation/` contains the core package: environments in `envs/`, RL code in `learning/`, shared helpers in `utils/`, `sensors/`, and `project_loading/`. Hydra config lives under `cfg/` with shared defaults in `cfg/base/`, experiments in `cfg/exp/`, and task definitions in `cfg/task/`. Runtime entry points are in `scripts/` (`train_ppo.py`, `eval.py`, `play.py`, `launch_ddp.sh`). Extension projects live in `projects/` and register through `pyproject.toml`. Composed robot MJCF is built with sibling **`aa-projects/assetx/`** (recipes → `artifacts/`); runtime bundles live under `.cache/aa-robot-models/` (`ROBOT_MODEL_DIR`), registered via `active_adaptation/assets/`. See `README.md` § Asset download and placement.

## Build, Test, and Development Commands
Install in a Python 3.11 environment with `pip install -e .`.
Use `aa-discover-projects` to refresh discovered project/task metadata, and `aa-list-tasks` to inspect available task IDs.
Typical workflows:

```bash
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
python scripts/eval.py task=Go2/Go2Flat algo=ppo eval_render=true
python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
bash scripts/launch_ddp.sh 0,1 train_ppo.py task=G1/G1LocoFlat algo=ppo
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for modules/functions, PascalCase for classes, and concise docstrings only where behavior is not obvious. Keep simple one-line operations inline instead of adding pass-through helper methods when inlining stays readable. Keep Hydra config keys and task names consistent with existing patterns such as `Go2/Go2Flat` and `ppo_symaug`. There is no pinned formatter in this repo today; keep imports grouped cleanly and match surrounding file structure. `pyproject.toml` enables Pyright checks, so prefer type-safe changes and preserve annotated APIs.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with focused runnable checks:

```bash
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo task.num_envs=16 wandb.mode=disabled
python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
pyright active_adaptation
```

Decision-value-driven experiments
1. Experiments should be driven by hypotheses and decision value. Do not run experiments just to fill out a table, complete a narrative, or make an ablation set look exhaustive; run them to answer a specific hypothesis or support a concrete decision.
2. Do not exhaustively explore directions with low marginal information gain. When a group of experiments is unlikely to improve performance, clarify direction, or inform follow-up work, especially when it would only reconfirm that an approach is infeasible or ineffective, stop instead of running exhaustive validation.

## Training Experiment Notes
When launching training remotely, follow these operational rules. For agent-oriented launch/monitor/kill workflows and `profiling.jsonl` sim perf tuning, see [.agents/skills/run-experiments/SKILL.md](.agents/skills/run-experiments/SKILL.md). HDMI-specific launch rules live in the HDMI project's `run-hdmi-experiments` skill.

- Before every launch, check live GPU occupancy with `nvidia-smi` and relevant `pgrep`/`tmux ls`; do not overlap GPUs with existing runs.
- Launch long runs in named `tmux` sessions and pipe stdout/stderr through `tee` into the experiment `records/` directory.
- After launch, do not assume success from process existence alone. Verify that the job passes asset resolution, environment creation, W&B initialization, and reaches the tqdm/iteration loop; record early W&B run ids and any restart reason in the markdown log.
- If a run hangs during startup, stop the tmux session and child `torchrun`/Python processes, archive the partial log, fix the root cause, and relaunch with a fresh log. Recheck that all GPUs are released before relaunching.

### Multi-GPU / DDP (Isaac + sensors)

`scripts/launch_ddp.sh 0,1 …` sets `CUDA_VISIBLE_DEVICES=0,1` for the whole `torchrun` job. In `aa.init(auto_rank=True)`, **`isolate_local_cuda_device()`** then remaps each rank to a **single** visible GPU **before** any CUDA / Isaac / Warp init:

| Rank | After isolation | Process-local device |
| --- | --- | --- |
| 0 | `CUDA_VISIBLE_DEVICES=0` | `cuda:0` (physical GPU 0) |
| 1 | `CUDA_VISIBLE_DEVICES=1` | `cuda:0` (physical GPU 1) |

**Why:** Isaac USDRT Fabric (`UsdStage::SelectPrims`) only supports process-local `cuda:0`. Without isolation, rank 1 uses `cuda:1` and hangs during sensor / `MeshRegistry` setup when Fabric pose reads run (e.g. pedestal `XformPrimView.get_world_poses()`), while rank 0 waits forever on the first NCCL collective.

**Agent rules for distributed code:**

- Use `aa.get_local_cuda_index()` (not `aa.get_local_rank()`) for Torch / Warp / DDP `device_ids` / NCCL `device=` after `aa.init`. Under isolation this is always `0`; `get_local_rank()` remains the process rank for logging and NCCL identity.
- Expect `cfg.device` / `env.device` to be `"cuda:0"` on **every** rank when isolated. Do not hard-code `cuda:{LOCAL_RANK}`.
- Call `aa.bind_local_rank_device()` after Kit / Warp work that may reset the current device, and immediately before NCCL collectives if isolation is not in effect. With CVD isolation it is mostly defensive (current device `0` is already correct).
- NCCL process group is created **after** `AppLauncher` (with `device_id` pinned to the local index). Do not assume `dist` is initialized before the Isaac app exists.
- CVD isolation is the default for one-env-per-rank DDP. Avoid it only when a single process must see multiple GPUs (model/pipeline parallel). Isolation must run before the first CUDA init.

Startup hang signatures to recognize:

- `SelectPrims: GPU N requested. GPUs other than cuda:0 are not currently supported` → missing / late CVD isolation; rank 1 stuck in sensor init.
- Rank 0 at `barrier` / `broadcast_object_list` with NCCL `Init START` and no rank-1 `sensor … ready` → same root cause (desync), not a broken NCCL install.

For new features, add the smallest reproducible command in the PR description and verify both config loading and the affected training/eval path.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `cleanup rewards` or `fix ground height query`. Keep commit subjects brief, lowercase, and specific to one change. PRs should explain the motivation, list the main files touched, include exact validation commands, and attach screenshots or videos for behavior/visualization changes. Link the relevant issue, experiment run, or WandB run when applicable.
