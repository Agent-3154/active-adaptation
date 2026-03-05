"""
This script is used to play and visualize a policy in the environment.
"""

import torch
import hydra
import itertools
import datetime
import copy
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.learning.modules.vecnorm import VecNorm

FILE_PATH = Path(__file__).parent


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu()
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"policy-{time_str}.onnx"
    export_onnx(deploy_policy, fake_input, str(path))


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=False)
    
    from helpers import make_env_policy
    env, policy = make_env_policy(cfg)
    
    if cfg.export_policy:
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_policy(env, policy, export_dir)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    carry = env.reset()
    
    assert not env.base_env.training

    vis_gs_rgb = cfg.get("vis_gs_rgb", False)
    vis_gs_rgb_interval = cfg.get("vis_gs_rgb_interval", 5)
    cv_window = None
    if vis_gs_rgb:
        try:
            import cv2
            cv2.namedWindow("GS RGB env0", cv2.WINDOW_NORMAL)
            cv_window = True
        except ImportError:
            print("[warn] vis_gs_rgb requires opencv-python, falling back to disabled")
            vis_gs_rgb = False

    dump_gs_imgs = cfg.get("dump_gs_imgs", False)
    dump_gs_imgs_interval = cfg.get("dump_gs_imgs_interval", 5)
    dump_gs_imgs_dir = cfg.get("dump_gs_imgs_dir", "gs_dump")
    if dump_gs_imgs:
        import os
        os.makedirs(dump_gs_imgs_dir, exist_ok=True)
        print(f"[dump_gs] Will save all-env GS images every {dump_gs_imgs_interval} steps -> {dump_gs_imgs_dir}")

    timer = Timer(env.step_dt)

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        # torch.compiler.cudagraph_mark_step_begin()
        
        for i in itertools.count():
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            # td_.update(td["next"])
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            if vis_gs_rgb and i % vis_gs_rgb_interval == 0:
                base = env.base_env if hasattr(env, "base_env") else env
                if hasattr(base, "debug_gs_render"):
                    rgb_np = base.debug_gs_render(env_id=0)
                    if rgb_np is not None:
                        import cv2
                        cv2.imshow("GS RGB env0", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            if dump_gs_imgs and i % dump_gs_imgs_interval == 0:
                base = env.base_env if hasattr(env, "base_env") else env
                if hasattr(base, "debug_dump_all_gs"):
                    base.debug_dump_all_gs(out_dir=dump_gs_imgs_dir, step=i)

            timer.sleep()
    
    env.close()


if __name__ == "__main__":
    main()

