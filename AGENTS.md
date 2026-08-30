# Repository Guidelines

## Project Structure & Module Organization
`active_adaptation/` contains the core package: environments in `envs/`, RL code in `learning/`, shared helpers in `utils/`, `sensors/`, and `project_loading/`. Hydra config lives under `cfg/` with shared defaults in `cfg/base/`, experiments in `cfg/exp/`, and task definitions in `cfg/task/`. Runtime entry points are in `scripts/` (`train_ppo.py`, `eval.py`, `play.py`, `launch_ddp.sh`). Extension projects live in `projects/` and register through `pyproject.toml`. Large robot and scene assets are expected under `.cache/aa-robot-models/`, not committed into the package.

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
When launching training remotely, follow these operational rules:

- Before every launch, check live GPU occupancy with `nvidia-smi` and relevant `pgrep`/`tmux ls`; do not overlap GPUs with existing runs.
- Use an experiment directory under `projects/hdmi/scripts/experiments/<date>_<name>/` for launch scripts, records, plots, and a live markdown log. Keep run keys, hosts, GPU ids, seeds, W&B ids, status, and exact commands in that log.
- For G1/HDMI runs that resolve assets through `mjhub.resolve_asset_reference`, set `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` unless intentionally refreshing the cache. If cache is missing, test asset resolution separately before launching the full DDP job; try host proxy port `7890` only for cache refresh, not as a default training dependency.
- When syncing to remote hosts with `sync-4090.sh` or `sync-h200.sh`, exclude experiment `records/` and `outputs/` directories before running `rsync --delete`, so live logs and generated plots are not wiped.
- Launch long runs in named `tmux` sessions and pipe stdout/stderr through `tee` into the experiment `records/` directory.
- After launch, do not assume success from process existence alone. Verify that the job passes asset resolution, environment creation, W&B initialization, and reaches the tqdm/iteration loop; record early W&B run ids and any restart reason in the markdown log.
- If a run hangs during startup, stop the tmux session and child `torchrun`/Python processes, archive the partial log, fix the root cause, and relaunch with a fresh log. Recheck that all GPUs are released before relaunching.
- For W&B-driven comparison runs, pull curves/plots only after the relevant group has enough iterations to answer the decision question; keep plot paths in the same experiment log.

For new features, add the smallest reproducible command in the PR description and verify both config loading and the affected training/eval path.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `cleanup rewards` or `fix ground height query`. Keep commit subjects brief, lowercase, and specific to one change. PRs should explain the motivation, list the main files touched, include exact validation commands, and attach screenshots or videos for behavior/visualization changes. Link the relevant issue, experiment run, or WandB run when applicable.
