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
Follow existing Python style: 4-space indentation, snake_case for modules/functions, PascalCase for classes, and concise docstrings only where behavior is not obvious. Keep Hydra config keys and task names consistent with existing patterns such as `Go2/Go2Flat` and `ppo_symaug`. There is no pinned formatter in this repo today; keep imports grouped cleanly and match surrounding file structure. `pyproject.toml` enables Pyright checks, so prefer type-safe changes and preserve annotated APIs.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with focused runnable checks:

```bash
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo task.num_envs=16 wandb.mode=disabled
python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
pyright active_adaptation
```

For new features, add the smallest reproducible command in the PR description and verify both config loading and the affected training/eval path.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `cleanup rewards` or `fix ground height query`. Keep commit subjects brief, lowercase, and specific to one change. PRs should explain the motivation, list the main files touched, include exact validation commands, and attach screenshots or videos for behavior/visualization changes. Link the relevant issue, experiment run, or WandB run when applicable.
