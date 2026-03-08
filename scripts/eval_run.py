import torch
import wandb
import os
import sys
import hydra
import argparse
from pathlib import Path
from active_adaptation.utils import wandb as aa_wandb_utils

from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from play import main as play_main
from eval import main as eval_main
from play_mujoco import main as play_mujoco_main

play = play_main.__wrapped__
eval = eval_main.__wrapped__
play_mujoco = play_mujoco_main.__wrapped__

FILE_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    parser.add_argument("-pm", "--play_mujoco", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    parser.add_argument("-o", "--teleop", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("-i", "--iterations", type=int, default=None)
    parser.add_argument("-l", "--lights", action="store_true", default=False,
                        help="Enable point lights in the scene")
    parser.add_argument("--vis-rgb", action="store_true", default=False,
                        help="Show GS-rendered RGB for env0 in a separate window")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override number of environments")
    args = parser.parse_args()

    api = wandb.Api()
    
    run = api.run(args.run_path)
    print(f"Loading run {run.name}")

    store_dir = aa_wandb_utils.get_store_dir()
    root = store_dir / run.name
    root.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    for file in run.files():
        print(file.name)
        if "checkpoint" in file.name:
            checkpoints.append(file)
        elif file.name == "cfg.yaml":
            file.download(str(root), replace=True)
            aa_wandb_utils._manifest_add_file(run, file.name, root / "cfg.yaml", kind="config")  # internal helper
        elif file.name == "files/cfg.yaml":
            file.download(str(root), replace=True)
            aa_wandb_utils._manifest_add_file(run, file.name, root / "cfg.yaml", kind="config")  # internal helper
        elif file.name == "config.yaml":
            file.download(str(root), replace=True)
            aa_wandb_utils._manifest_add_file(run, file.name, root / "config.yaml", kind="config")  # internal helper

    # `run.config` does not preserve order of the keys
    # so we need to manually load the config file :(
    # if os.path.exists(os.path.join(root, "config.yaml")):
    #     cfg = OmegaConf.load(os.path.join(root, "config.yaml"))
    #     for k, v in run.config.items():
    #         cfg[k] = cfg[k]["value"]
    # else:
    try:
        cfg = OmegaConf.load(os.path.join(root, "files", "cfg.yaml"))
    except FileNotFoundError:
        cfg = OmegaConf.load(os.path.join(root, "cfg.yaml"))
    OmegaConf.set_struct(cfg, False)

    if args.iterations is not None:
        cfg["checkpoint_path"] = f"run:{args.run_path}:{args.iterations}"
    else:
        cfg["checkpoint_path"] = f"run:{args.run_path}"
    cfg["vecnorm"] = "eval"
    # cfg["algo"]["phase"] = "adapt"
    # cfg['algo']["phase"] = "finetune"
    if args.teleop:
        cfg["task"]["command"]["teleop"] = True

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        # cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        if args.terrain:
            cfg["task"]["terrain"] = _cfg.task.terrain
        if args.command:
            cfg["task"]["command"] = _cfg.task.command
    
    if args.lights:
        cfg["task"]["scene"]["point_lights"] = True
    if args.play:
        cfg["task"]["command"]["vis_waypoints"] = True
    if args.vis_rgb:
        cfg["vis_gs_rgb"] = True
        cfg["vis_gs_rgb_interval"] = 5

    num_envs = args.num_envs or 1

    assert not (args.play and args.play_mujoco), "Cannot play and play_mujoco at the same time"
    if args.play:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = num_envs
        cfg["task"]["scene"]["n_repeats"] = num_envs
        cfg["export_policy"] = args.export
        play(cfg)
    elif args.play_mujoco:
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = num_envs
        cfg["task"]["scene"]["n_repeats"] = num_envs
        cfg["export_policy"] = args.export
        play_mujoco(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 16
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            # cfg["app"]["headless"] = False
        eval(cfg)


if __name__ == "__main__":
    main()