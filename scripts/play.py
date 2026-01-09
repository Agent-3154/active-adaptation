"""
This script is used to play and visualize a policy in the environment.
"""

import torch
import hydra
import itertools
import datetime
import copy
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
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

    if cfg.get("device", "auto") == "auto":
        cfg.device = "cuda" if aa.get_backend() == "mjlab" else "cpu"
        print(f"Using device: {cfg.device}")
    
    aa.init(cfg, auto_rank=True, import_projects=True)
    
    from helpers import EpisodeStats, make_env_policy
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
            
            timer.sleep()
    
    env.close()


if __name__ == "__main__":
    main()

