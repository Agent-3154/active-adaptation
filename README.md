# active-adaptation

`active-adaptation` is a fast-moving, research-oriented RL codebase for various robotic tasks and algorithms. It emphasizes environment flexibility and ease of use for research.

Projects using this codebase:

* [FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots](https://arxiv.org/abs/2505.06883)

* [HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos](https://arxiv.org/abs/2509.16757)

* [GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction](https://arxiv.org/abs/2511.04679)

* [Gallant: Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains](https://arxiv.org/abs/2511.14625)

* [MimicLite](https://github.com/EGalahad/mimic-lite) at branch `dev/hdmi`.

and more to come...

Note: The main branch is fast-moving and therefore **almost constantly broken somewhere**. We intentionally practice human design and limit AI-generated code. The best usage of this codebase is to read it instead of directly using it for your own projects.

## Installation

### Workspace layout

For IsaacLab development, the recommended workspace layout is:

```bash
${workspaceFolder}/
  .vscode/
    launch.json
    settings.json
  active-adaptation/
  IsaacLab/
    _isaac_sim/
```

### Recommended: uv multi-environment workflow

Use `uv` as the default environment manager. Keep backend stacks isolated:

- `venv/isaac51`: Python `==3.11.*`
- `venv/isaac60`: Python `==3.12.*`
- `venv/mjlab`: Python `>=3.11`

Setup:

```bash
git clone git@github.com:btx0424/active-adaptation.git
cd active-adaptation

# shared tooling / backend-agnostic environment
# you may `uv python pin 3.11` to share uv's cache across backends
uv sync

# backend-specific environments
uv sync --project venv/isaac51
# uv sync --project venv/isaac60 # not supported yet
uv sync --project venv/mjlab
```

### IsaacLab Installation

We will install [IsaacLab](https://github.com/isaac-sim/IsaacLab) from source, not from pip.

> `isaac51` is currently the only tested Isaac track. `isaac60` setup is planned but not validated yet.

Manual installation steps (`isaac51`):

```bash
# from active-adaptation repo
cd /path/to/active-adaptation
uv sync --project venv/isaac51
source venv/isaac51/.venv/bin/activate

# install IsaacLab extensions from source repo
cd /path/to/IsaacLab
./isaaclab.sh -i none

# optional: verify IsaacLab import in this env
python -c "import isaaclab; print(isaaclab.__file__)"
```

Common commands:

```bash
# shared
uv run aa-discover-projects
uv run aa-list-tasks
uv run pyright active_adaptation

# backend-specific runs
uv run --project venv/isaac51 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
# uv run --project venv/isaac60 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo # not supported yet
uv run --project venv/mjlab scripts/train_ppo.py task=Go2/Go2Flat algo=ppo backend=mjlab

# multi-GPU helper (DDP via torchrun)
uv run --project venv/isaac51 ./scripts/launch_ddp.sh 0,1 scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
```

Notes:

- Prefer `uv run --project <env-dir>` for reproducible backend runs.
- Use `uv run --with <extra> ...` only for temporary one-off tools, not core backend dependencies.
- Keep backend-specific `warp-lang` pins in each backend env (`venv/isaac51`, `venv/isaac60`, `venv/mjlab`), not in the root project.

### Conda workflow (supported, legacy recommendation)

Conda remains supported if you prefer it for Python/runtime management. The same split-env principle applies (do not combine incompatible backends in one env).

```bash
git clone git@github.com:btx0424/active-adaptation.git
cd active-adaptation

# isaac51 (Python 3.11)
conda create -n aa-isaac51 python=3.11 -y
conda activate aa-isaac51
pip install -e .
# install isaac51-specific deps (including its warp-lang pin)

# isaac60 (Python 3.12)
conda create -n aa-isaac60 python=3.12 -y
conda activate aa-isaac60
pip install -e .
# install isaac60-specific deps (including its warp-lang pin)

# mjlab (Python >=3.11, example with 3.11)
conda create -n aa-mjlab python=3.11 -y
conda activate aa-mjlab
pip install -e .
pip install mjlab
```

If you use conda, prefer one env per backend track and document exact backend package pins in your team setup docs.

After that, run training / eval / play from the repo root:

```bash
uv run --project venv/isaac51 python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
uv run --project venv/isaac51 python scripts/eval.py task=Go2/Go2Flat algo=ppo eval_render=true
uv run --project venv/isaac51 python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
```

Notes:

- `uv sync --project venv/isaac51` manages the tested Isaac track dependencies.
- IsaacLab itself may still require its own setup for `PYTHONPATH`, Isaac Sim linking, and extension discovery.
- The important constraint is that IsaacLab and this repo must use the same `venv/isaac51` environment.

### MJLab setup

If you want to use the `mjlab` backend, use the dedicated `venv/mjlab` environment:

```bash
uv sync --project venv/mjlab
```

Then run MJLab commands from the repo root:

```bash
uv run --project venv/mjlab python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo backend=mjlab
uv run --project venv/mjlab python scripts/play.py task=Go2/Go2Flat algo=ppo backend=mjlab checkpoint_path=/path/to/checkpoint.pt
```

### Optional VSCode setup

Edit `.vscode/settings.json` on demand:

```json
"python.analysis.extraPaths": [
  "./IsaacLab/source/isaaclab",
  "./IsaacLab/source/isaaclab_assets",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior.ui",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.domain_randomization",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.examples",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.scene_blox",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.synthetic_recorder",
  "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.writers"
],
```

## Asset download and placement

Some robots and scene files are **not** shipped inside this repository (to keep the clone small). They are loaded from a fixed cache directory next to the package.

### Where files must live

After `pip install -e .`, the code resolves assets from:

**`<active-adaptation repo root>/.cache/aa-robot-models/`**

That path is `ROBOT_MODEL_DIR` in code (`CACHE_DIR` is the repo’s `.cache/` folder). Do not rename `aa-robot-models` unless you also change the code.

### What to download

- **Source:** [Hugging Face dataset `btx0424/aa-robot-models`](https://huggingface.co/datasets/btx0424/aa-robot-models)
- **Layout under `aa-robot-models/`** (paths used today):
  - `a2/` — Unitree A2 MJCF/USD (`a2.xml`, `a2.usd`)
  - `b2/` — Unitree B2 MJCF/USD (`b2.xml`, `b2_flattened.usda`)
  - `scene/` — e.g. `kloofendal_43d_clear_puresky_4k.hdr` (dome light / sky for the Isaac backend)

If the archive or clone has an extra top-level folder, unpack or move contents so those directories sit **directly** under `.cache/aa-robot-models/`.

### How to get them

From the **root of the cloned `active-adaptation` repo** (where `.cache/` is created automatically):

```bash
# Option A: Hugging Face CLI (recommended)
pip install -U "huggingface_hub[cli]"
huggingface-cli download btx0424/aa-robot-models --repo-type dataset --local-dir .cache/aa-robot-models
```

You can instead **clone or copy** the dataset contents into `.cache/aa-robot-models/`, or put the data elsewhere and replace `.cache/aa-robot-models` with a **symlink** to that folder.

## CLI commands

These commands are available after `pip install -e .` and help manage projects and tasks.

| Command | Description |
|--------|-------------|
| `aa-create-project` | Create a new active-adaptation project scaffold. |
| `aa-discover-projects` | Discover installed projects and learning modules, write/update `projects.json`. |
| `aa-list-tasks` | List task names from `cfg/task` in active-adaptation and discovered projects. |
| `aa-pull` | Run `git pull` for active-adaptation and all enabled projects. |
| `aa-recent-commands` | List recent training/eval commands from stored history. |

### aa-create-project

Create a new project with packages `{name}/` and `{name}_learning/`, `pyproject.toml`, `cfg/task`, `cfg/exp`, and optional README/`.gitignore` (existing files are not overwritten, e.g. when scaffolding inside a new git repo).

```bash
aa-create-project -n myproject
aa-create-project -n myproject -d /path/to/parent
```

- **`-n`, `--name`** (required): Project/package name (lowercase, alphanumeric + underscores).
- **`-d`, `--dir`**: Parent directory for the new project folder (default: current directory).

### aa-discover-projects

Scans entry points `active_adaptation.projects` and `active_adaptation.learning` and updates `projects.json` (under the cache directory) with project paths and task dirs. Use this after installing or adding projects so that `aa-list-tasks` and `aa-pull` know about them. Edit `projects.json` to enable or disable projects.

```bash
aa-discover-projects
```

### WandB defaults in projects.json

You can set default WandB settings in `.cache/projects.json` so training scripts pick them up automatically during `aa.init(...)`.

Supported keys:

- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`

You can define these either:

1. In a top-level `wandb` block (global defaults), or
2. Inside enabled `environment` project entries (project-scoped defaults).

Example:

```json
{
  "environment": {
    "hoi1": {
      "enabled": true,
      "WANDB_ENTITY": "G1_Hoi",
      "WANDB_PROJECT": "object-hoi"
    }
  }
}
```

Resolution behavior:

- `WANDB_API_KEY` and `WANDB_ENTITY` are applied as environment defaults only when those env vars are not already set.
- `WANDB_PROJECT` overrides `cfg.wandb.project` when configured in `projects.json`.
- If no manifest defaults are configured, WandB initializes normally:
  - entity comes from env vars or your global WandB settings
  - project comes from Hydra config (`cfg.wandb.project`)
- If multiple enabled projects define conflicting values for the same key, that key is ignored and a warning is logged.

### aa-list-tasks

Prints task IDs from YAML files under `cfg/task` for active-adaptation and for each enabled project in `projects.json`. Task names keep the directory prefix (e.g. `G1/G1LocoFlat`). Useful to see which tasks are available for `task=...` in training/eval.

```bash
aa-list-tasks
```

### aa-pull

Runs `git pull` in the active-adaptation repo and in all **enabled** projects listed in `projects.json`. Use after `aa-discover-projects` so projects are registered.

```bash
aa-pull           # active projects only
aa-pull --all     # all discovered projects, including disabled
```

### aa-recent-commands

Shows the last N commands (training/eval runs) from the stored command history. Optional filter by script name.

```bash
aa-recent-commands
aa-recent-commands -n 10
aa-recent-commands -s train_ppo -s eval_run
```

- **`-n`, `--num`**: Number of recent commands (default: 5).
- **`-s`, `--script`**: Filter by script name (e.g. `train_ppo`, `eval_run`); can be repeated (OR).


## Basic Usage

### Training

Examples:

```bash
python test_env.py task=Go2/Go2Flat algo=ppo
# hydra command-line overrides
python test_env.py task=Go2/Go2Flat algo=ppo algo.entropy_coef=0.002 total_frames=200_000_000 task.terrain=medium
# finetuning
python test_env.py task=Go2/Go2Flat algo=ppo checkpoint_path=${local_checkpoint_path}
python test_env.py task=Go2/Go2Flat algo=ppo checkpoint_path=run:${wandb_run_path}
# multi-GPU training
export OMP_NUM_THREADS=4 # a number greater than 1
python -m torch.distributed --nnodes=1 --nproc-per-node=4 ...
```

### VSCode/Cursor Python Debugging

Create and modify `.vscode/launch.json` to add debug configurations. For example:
```json
"configurations": [
  {
      "name": "Python Debugger: Go2 Loco",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {"CUDA_VISIBLE_DEVICES": "0"},
      "args": [
          "task=Go2/Go2Force",
          "algo=ppo_dic_train",
          "algo.symaug=True",
          "wandb.mode=disabled",
          "task.num_envs=16"
      ]
  }
]
```
