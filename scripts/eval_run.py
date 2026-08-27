import os
import argparse
import hydra
from active_adaptation.utils import wandb as aa_wandb_utils

from omegaconf import OmegaConf
from play import main as play_main
from eval import main as eval_main

play = play_main.__wrapped__
eval = eval_main.__wrapped__

FILE_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    parser.add_argument("-pm", "--play_mujoco", action="store_true", default=False)
    parser.add_argument("-pl", "--play_mjlab", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    parser.add_argument("-o", "--teleop", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("-i", "--iterations", type=int, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        default=False,
        help="Force a W&B API refresh even when a finished run is cached locally.",
    )
    args = parser.parse_args()

    resolved = aa_wandb_utils.resolve_wandb_run(
        args.run_path,
        args.iterations,
        force=args.refresh,
    )
    print(f"Loading run {resolved.name}")

    cfg = OmegaConf.load(resolved.cfg_path)
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
    
    play_modes = sum([args.play, args.play_mujoco, args.play_mjlab])
    assert play_modes <= 1, "Use at most one of --play, --play_mujoco, --play_mjlab"
    if args.play:
        # this will use the original backend during training
        cfg["headless"] = False
        cfg["app"]["headless"] = False
        cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        play(cfg)
    elif args.play_mujoco:
        cfg["backend"] = "mujoco"
        cfg["headless"] = False
        cfg["task"]["num_envs"] = 1
        cfg["export_policy"] = args.export
        play(cfg)
    elif args.play_mjlab:
        cfg["backend"] = "mjlab"
        cfg["headless"] = False
        cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        play(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 16
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            # cfg["app"]["headless"] = False
        eval(cfg)


if __name__ == "__main__":
    main()
