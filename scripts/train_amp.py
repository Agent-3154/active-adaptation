import torch
import hydra
import wandb
import datetime
from setproctitle import setproctitle
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from torchrl.envs import TransformedEnv, Compose, InitTracker, StepCounter
from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.wandb import parse_checkpoint_path
from active_adaptation.learning.mimic.discriminator import Discriminator, DiscriminatorCfg


@hydra.main(config_path="../cfg", config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, import_projects=True)        

    from active_adaptation.envs.locomotion import (
        SimpleEnvIsaac,
        SimpleEnvMujoco,
        SimpleEnvMjlab
    )
    EnvClass = {
        "isaac": SimpleEnvIsaac,
        "mujoco": SimpleEnvMujoco,
        "mjlab": SimpleEnvMjlab,
    }[cfg.backend]

    env = EnvClass(cfg.task, str(cfg.device), headless=cfg.headless)
    env = TransformedEnv(env, Compose(InitTracker(), StepCounter()))

    discriminator = Discriminator(
        cfg.model,
        device=torch.device(cfg.device)
    )

    PolicyClass = hydra.utils.get_class(cfg.algo._target_)
    policy = PolicyClass(
        cfg=cfg.algo,
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        reward_spec=env.reward_spec,
        device=env.device,
        env=env
    )
    rollout_policy = policy.get_rollout_policy("train")
    if (checkpoint_path := parse_checkpoint_path(cfg.checkpoint_path)) is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
        policy.load_state_dict(state_dict["policy"])
        print(f"Loaded policy from checkpoint: {checkpoint_path}")

    # rollout_policy = discriminator.play_motion
    
    # if (checkpoint_path := parse_checkpoint_path(cfg.checkpoint_path)) is not None:
    #     state_dict = torch.load(checkpoint_path, weights_only=False)
    #     policy.load_state_dict(state_dict["policy"])
    #     print(f"Loaded policy from checkpoint: {checkpoint_path}")
    
    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()
        
        default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

    from helpers import EpisodeStats
    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    print(f"Log interval: {log_interval} steps")

    def save(policy, checkpoint_name: str):
        ckpt_path = Path(run.dir) / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        torch.save(state_dict, ckpt_path)
        # run.save(ckpt_path, policy="now", base_path=run.dir)
        return ckpt_path
    
    @torch.inference_mode()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(carry):
        data = []
        for _ in range(32):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            private_keys = [key for key in td.keys(True, True) if isinstance(key, str) and key.startswith('_')]
            td = td.exclude(*private_keys)
            data.append(td.to(policy.device))
        data = torch.stack(data, dim=1)
        return data, carry
    
    carry = env.reset()
    ckpt_path = None
    for i in tqdm(range(1000), disable=not aa.is_main_process()):
        data, carry = collect(carry)
        episode_stats.add(data)

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        info.update(env.stats_ema)

        info.update(discriminator.train_op(data))
        info.update(policy.train_op(data))

        if aa.is_main_process():
            if i > 0 and i % cfg.save_interval == 0:
                ckpt_path = save(policy, f"checkpoint_{i}")
            
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
            print(f"Latest checkpoint: {ckpt_path}")
            run.log(info)
    
    if aa.is_main_process():
        ckpt_path = save(policy, "checkpoint_final")
        print(f"Final checkpoint: {ckpt_path}")
        wandb.finish()


if __name__ == "__main__":
    main()
    