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

1. For the following steps, the recommended way to structure the (VSCode or Cursor) workspace is:
   ```bash
    ${workspaceFolder}/ # File->Open Folder here
      .vscode/
        launch.json # use vscode Python debugging for better experience!
        settings.json
      active-adaptation/
      IsaacLab/
        _isaac_sim/
   ```
2. Install [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) by downloading the latest release and unzip it to a desired location `$ISAACSIM_PATH`. Installing via `pip` is also fine.
3. Install [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and setup a conda environment:
   ```bash
   conda create -n lab python=3.11
   conda activate lab
   # install IsaacLab to the exisiting conda environment
   # git clone https://github.com/isaac-sim/IsaacLab.git
   git clone git@github.com:isaac-sim/IsaacLab.git # SSH recommended
   cd IsaacLab
   ln -s $ISAACSIM_PATH _isaac_sim
   ./isaaclab.sh -c lab
   ./isaaclab.sh -i none # install without additional RL libraries
   # reactivate the environment
   conda activate lab
   echo $PYTHONPATH 
   ```
   You should see the isaac-sim related dependencies are added to `$PYTHONPATH`.
4. [**Recommended**] Isaac Sim comes with its cumstom Python environment which may lead to conflicts with our conda environment.
   To avoid Python environment conflicts, try the following steps:
   ```bash
   cd $ISAACSIM_PATH/exts/omni.isaac.ml_archive
   mv pip_prebundle pip_prebundle.back # backup the packages shipped with Isaac Sim
   ln -s $CONDA_PREFIX/lib/python3.11/site-packages pip_prebundle
   ```
5. [**Optional**] VSCode setup. This enables the Python extension for code analysis to provide auto-completiong and linting. Edit `.vscode/settings.json` on demand:
   ```json
   "python.analysis.extraPaths": [
        // Recommended
        "./IsaacLab/source/isaaclab",
        "./IsaacLab/source/isaaclab_assets",
        // Optional, modified from IsaacLab/.vscode/settings.json
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior.ui",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.domain_randomization",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.examples",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.scene_blox",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.synthetic_recorder",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.writers",
        //... note that adding extraPaths may increase VSCode CPU usage
    ],
   ```
6. `pip install -U torch torchvision tensordict torchrl`. `torch>=2.8.0` is recommended for full feature.
7. Install this repo:
   ```bash
   git clone git@github.com:btx0424/active-adaptation.git # SSH recommended
   cd active-adaptation
   pip install -e . 
   ```


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