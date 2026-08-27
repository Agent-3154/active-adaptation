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

## Table of contents

- [Installation](#installation)
  - [Workspace layout](#workspace-layout)
  - [Recommended: uv multi-environment workflow](#recommended-uv-multi-environment-workflow)
  - [IsaacLab Installation](#isaaclab-installation)
  - [Conda workflow (supported, legacy recommendation)](#conda-workflow-supported-legacy-recommendation)
  - [MJLab setup](#mjlab-setup)
  - [Optional VSCode setup](#optional-vscode-setup)
- [Asset download and placement](#asset-download-and-placement)
- [Project management](#project-management)
  - [Why packaging / entry points](#why-packaging--entry-points)
  - [Project layout](#project-layout)
  - [Create a project](#create-a-project)
  - [Install a project](#install-a-project)
  - [Discover and enable](#discover-and-enable)
  - [Pull updates](#pull-updates)
  - [How loading works at runtime](#how-loading-works-at-runtime)
  - [WandB defaults in `projects.json`](#wandb-defaults-in-projectsjson)
  - [CLI reference](#cli-reference)
- [Basic Usage](#basic-usage)
  - [Training](#training)
  - [VSCode/Cursor Python Debugging](#vscodecursor-python-debugging)

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

Extension projects (tasks, MDP terms, custom algos) usually live as sibling repos, e.g. `aa-projects/<name>/`, and are registered through Python packaging — see [Project management](#project-management).

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
uv run aa-project discover
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

## Project management

Extension projects add task YAMLs, MDP terms, assets, and/or learning algorithms on top of `active-adaptation`. They are separate Python packages, discovered via packaging entry points, then selectively imported at runtime.

Typical flow:

1. **Create** a local scaffold, or **install** from a GitHub URL (clone + editable install).
2. Ensure the project is installed into the **backend** env you train with (`venv/isaac51`, `venv/mjlab`, …).
3. **Discover** entry points into `.cache/projects.json`.
4. **Enable** the project(s) you want loaded.
5. Run training / eval — `aa.init` imports enabled packages and Hydra picks up their `cfg/`.

### Why packaging / entry points

Projects register through:

| Entry-point group | Role |
|-------------------|------|
| `active_adaptation.projects` | Environment / MDP package (commands, rewards, assets, …) |
| `active_adaptation.learning` | Algo configs / policies |

We use packaging (instead of only scanning a folder like `aa-projects/`) so that **installing a project also installs its dependencies** into the same backend environment. Path-only loading would put code on `sys.path` but leave third-party deps missing.

Install projects into the **backend** env you actually run (`uv run --project venv/isaac51 …`), not only the shared root env. Do not dump project-specific deps into the root `active-adaptation` `pyproject.toml`.

### Project layout

Scaffolded projects look like:

```text
myproject/
├── cfg/
│   ├── task/          # Hydra task YAMLs
│   └── exp/           # Optional experiment overlays
├── src/
│   ├── myproject/           # Env / MDP package
│   └── myproject_learning/  # Algo configs / policies
└── pyproject.toml           # entry points + dependencies
```

Built-in examples also live under this repo’s `projects/` (e.g. `facet`, `mimic`, `metamorph`) and register from the root `pyproject.toml`.

### Create a project

```bash
# run inside the backend env you train with
uv run --project venv/isaac51 aa-project create -n myproject
# -> scaffolds <workspace>/aa-projects/myproject
# -> editable-installs into the current env (--no-deps by default)
# -> runs discover

aa-project create -n myproject -d /path/to/parent
# -> /path/to/parent/myproject (+ install + discover)
```

- **`-n`, `--name`** (required): lowercase alphanumeric + underscores. Creates packages `src/{name}/` and `src/{name}_learning/`.
- **`-d`, `--dir`**: parent directory for the new project folder (default: sibling `aa-projects/` next to the `active-adaptation` repo).
- **`--no-deps` / `--deps`**: editable-install without (default) or with dependency resolution.
- **`--skip-discover`**: scaffold + install only; do not refresh `.cache/projects.json`.

The scaffold writes `pyproject.toml` with both entry-point groups, `cfg/task`, `cfg/exp`, README, and `.gitignore` (existing README/`.gitignore` are kept). New projects still start as **disabled** in `projects.json` until you `aa-project enable <name>`.

Declare any project-specific third-party packages in that project’s `[project.dependencies]` (in addition to `active_adaptation`).

### Install a project

**From GitHub** (clone into `--dir`, editable-install into the *current* interpreter, then discover):

```bash
# run inside the backend env you train with
uv run --project venv/isaac51 aa-project install git@github.com:ORG/myproject.git -d ../aa-projects
uv run --project venv/isaac51 aa-project install https://github.com/ORG/myproject.git -d ../aa-projects --no-deps
```

- **`URL`**: HTTPS or SSH GitHub URL.
- **`-d`, `--dir`**: parent directory for `git clone` (default: `.`).
- **`--no-deps`**: install the project package only (avoids re-resolving transitive deps that can disturb a locked Isaac/mjlab env when `active_adaptation` is already present).
- **`--skip-discover`**: do not refresh `.cache/projects.json` after install.

**Already cloned locally:**

```bash
cd /path/to/myproject
uv pip install -e . --python /path/to/active-adaptation/venv/isaac51/.venv/bin/python
# or with the backend env activated:
pip install -e .
# then:
uv run --project venv/isaac51 aa-project discover
```

Repeat for each backend env that should see the project. Prefer `--no-deps` when the backend env already has `active_adaptation` and you only need the project’s own extra packages installed separately.

### Discover and enable

```bash
uv run --project venv/isaac51 aa-project discover
```

`aa-project discover` scans installed entry points and updates `.cache/projects.json` with package paths, task dirs, and an `enabled` flag (new entries default to disabled unless `--enabled` is passed). Then enable what you need:

```bash
aa-project enable myproject
aa-project disable myproject
aa-project enable          # enable all discovered projects
aa-project disable         # disable all discovered projects
```

The same entry-point name may appear under both `environment` and `learning`; enable/disable updates both when present. You can also edit `.cache/projects.json` by hand.

List tasks after discovery:

```bash
aa-list-tasks
```

### Pull updates

```bash
aa-project pull                 # active-adaptation + enabled projects
aa-project pull --all           # active-adaptation + all discovered projects
aa-project pull myproject       # one project only
```

Run `aa-project discover` again after adding or relocating installs so paths stay current.

### How loading works at runtime

When you run a script that calls `aa.init(...)`:

1. **Environment packages** listed as `enabled` in `projects.json` are imported (MDP terms / assets register as a side effect).
2. Hydra’s search-path plugin appends each enabled project’s `cfg/` directory and imports enabled **learning** modules (algo `ConfigStore` registration).

Disabled projects are ignored. Only enable projects you need for a run — imports have side effects and can collide on class names.

### WandB defaults in `projects.json`

You can set default WandB settings in `.cache/projects.json` so training scripts pick them up during `aa.init(...)`.

Supported keys: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`.

Define them either:

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

- `WANDB_API_KEY` and `WANDB_ENTITY` apply as environment defaults only when those env vars are not already set.
- `WANDB_PROJECT` overrides `cfg.wandb.project` when configured in `projects.json`.
- If no manifest defaults are set, WandB initializes normally (entity from env / global settings; project from Hydra).
- If multiple enabled projects define conflicting values for the same key, that key is ignored and a warning is logged.

### CLI reference

Available after installing `active-adaptation` (root `uv sync` or `pip install -e .`).

| Command | Description |
|--------|-------------|
| `aa-project create -n NAME [-d DIR] [--deps] [--skip-discover]` | Scaffold, editable-install, and discover (default: sibling `aa-projects/<name>`). |
| `aa-project install URL [-d DIR] [--no-deps] [--skip-discover]` | Clone from GitHub and editable-install into the current env. |
| `aa-project discover [--enabled]` | Scan installed entry points; update `.cache/projects.json`. |
| `aa-project enable [NAME]` | Enable one project, or all if `NAME` is omitted. |
| `aa-project disable [NAME]` | Disable one project, or all if `NAME` is omitted. |
| `aa-project pull [NAME] [--all]` | `git pull` for active-adaptation and/or projects. |
| `aa-list-tasks` | List task IDs from `cfg/task` in active-adaptation and discovered projects. |

```bash
aa-project --help
aa-project create --help
```

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
