# active-adaptation

<!-- ## Note (2025.8.26)

Thanks for taking a look! The code base was shared for multiple projects so it contains some old code that are no-longer usable. We are actively working on **cleaning up and refactoring to make it camera-ready** (e.g, compatible with Isaac Sim 5.0). It will be ready by the date of CoRL 2025. The core implementation of our CoRL paper [FACET](https://arxiv.org/abs/2505.06883) can be found at `active_adaptation/envs/mdp/commands/facet_commands`.

Meanwhile, the code for the live demo (runinng Mujoco in browsers) is here [https://github.com/Facet-Team/facet]. -->

## Features
* Automatic shape handling for observation.
* Clean and efficient single-file RL implementation.
* Easy symmetry augmentation.
* Seamless Mujoco sim2sim.

Projects using this codebase:

* [FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots](https://arxiv.org/abs/2505.06883)

* [HDMI: Learning Interactive Humanoid Whole-Body Control from * Human Videos](https://arxiv.org/abs/2509.16757)

* [GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction](https://arxiv.org/abs/2511.04679)

* [Gallant: Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains](https://arxiv.org/abs/2511.14625)

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

### Single Python environment

Use one Python 3.11 environment for this repo.

- Isaac Sim / IsaacLab requires Python 3.11.
- MJLab also works on Python 3.11.
- The repo root is the only `uv` project you need.

Setup:

```bash
git clone git@github.com:btx0424/active-adaptation.git
cd active-adaptation
uv sync
```

For common development commands:

```bash
uv run aa-discover-projects
uv run aa-list-tasks
uv run pyright active_adaptation
```

### IsaacLab / Isaac Sim setup

Install [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) and [IsaacLab](https://github.com/isaac-sim/IsaacLab) into the same Python 3.11 environment you use for this repo.

After that, run training / eval / play from the repo root:

```bash
uv run python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
uv run python scripts/eval.py task=Go2/Go2Flat algo=ppo eval_render=true
uv run python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
```

Notes:

- `uv sync` manages this repo's dependencies in `.venv`.
- IsaacLab itself may still require its own setup for `PYTHONPATH`, Isaac Sim linking, and extension discovery.
- The important constraint is that IsaacLab and this repo must use the same Python 3.11 environment.

### MJLab setup

If you want to use the `mjlab` backend, install the optional extra in the same root environment:

```bash
uv sync --extra mjlab
```

Then run MJLab commands from the repo root:

```bash
uv run python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo backend=mjlab
uv run python scripts/play.py task=Go2/Go2Flat algo=ppo backend=mjlab checkpoint_path=/path/to/checkpoint.pt
```

### Optional: direct `pip` install

If you are not using `uv`, the root package can still be installed directly:

```bash
pip install -e .
pip install mjlab==1.2.0  # only if you need the mjlab backend
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
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
# hydra command-line overrides
python scripts/train_ppo.py task=Go2/Go2Flat task.num_envs=8192 algo=ppo algo.entropy_coef=0.002 total_frames=200_000_000
# finetuning
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo checkpoint_path=${local_checkpoint_path}
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo checkpoint_path=run:${wandb_run_path}
# train using mjlab (requires mjwarp and mjlab)
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo backend=mjlab
# single-node multi-GPU training
export OMP_NUM_THREADS=4 # a number greater than 1
bash scripts/launch_ddp.sh 0,1,2,3 train_ppo.py task=G1/G1LocoFlat ...
```

### Evaluation and Visualization

Examples:

```bash
# play the policy
python play.py task=Go2/Go2Flat algo=ppo checkpoint_path=${local_checkpoint_path}
python play.py task=Go2/Go2Flat algo=ppo checkpoint_path=run:${wandb_run_path}
# play with mujoco backend
python play.py task=Go2/Go2Flat algo=ppo backend=mujoco
# export to onnx for deployment
python play.py task=Go2/Go2Flat algo=ppo export_policy=true
# record video
python eval.py task=Go2/Go2Flat algo=ppo eval_render=true
# coordination with servers or other collaborators
python eval_run.py --run_path ${wandb_run_path} --play # eval/visualize remote runs
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
