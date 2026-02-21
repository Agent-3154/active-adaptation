import torch
import hydra
import itertools
import datetime
from pathlib import Path
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation
from active_adaptation.utils.export import export_onnx


active_adaptation.import_algorithms()
FILE_PATH = Path(__file__).parent


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    from helpers import EpisodeStats, make_env_policy
    env, policy = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        import copy
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()
        print(f"Inference time of policy: {test(deploy_policy, fake_input)}")

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f"policy-{time_str}.onnx"

        export_onnx(deploy_policy, fake_input, str(path))

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    carry = env.reset()
    assert not env.base_env.training
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        torch.compiler.cudagraph_mark_step_begin()
        for i in itertools.count():
            carry = policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            # td_.update(td["next"])
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

