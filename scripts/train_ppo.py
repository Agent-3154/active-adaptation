import torch
import hydra
import numpy as np
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig

from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase

import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.ppo_base import PPOBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        os.makedirs(run.dir, exist_ok=True)
        cfg_save_path = os.path.join(run.dir, "cfg.yaml")
        OmegaConf.save(cfg, cfg_save_path)
        run.save(cfg_save_path, policy="now")
        run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    from helpers import make_env_policy, evaluate
    from active_adaptation.utils.helpers import EpisodeStats

    env, policy = make_env_policy(cfg)
    policy: PPOBase

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        torch.save(state_dict, ckpt_path)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        return ckpt_path

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and i % save_interval == 0

    ckpt_path = None
    carry = env.reset()
    observation_keys = list(env.observation_spec.keys(True, True))

    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(carry, rollout_policy: TensorDictModuleBase, transitions: bool=False):
        """
        If transitions is True, we collect and store transitions.
        If transitions is False, we store only the trajectories. Due to autoreset, we do not have
        the next states at terminal states. In this case, we approximate V_{t+1} with V_t.
        """
        data = []
        for _ in range(cfg.algo.train_every):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            private_keys = [
                key
                for key in td.keys(True, True)
                if isinstance(key, str) and key.startswith("_")
            ]
            td = td.exclude(*private_keys)
            if not transitions:
                td["next"] = td["next"].exclude(*observation_keys)

            data.append(td.to(policy.device))
        data = torch.stack(data, dim=1)

        if not transitions:
            state_value = data["state_value"]
            next_state_value = policy.compute_value(carry.copy())["state_value"]
            next_state_value = torch.cat([
                data["state_value"][:, 1:],
                next_state_value.unsqueeze(1),
            ], dim=1)
            # since we have not stored the terminal states, we approximate V_{t+1} with V_t
            data["next", "state_value"] = torch.where(
                data["next", "done"],
                state_value,
                next_state_value,
            )
        return data, carry

    env_frames = 0

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    for stage in stages:

        rollout_policy = policy.get_rollout_policy("train")
        policy.on_stage_start(stage)

        if aa.is_main_process():
            progress = tqdm(range(total_iters), desc=stage)
        else:
            progress = range(total_iters)

        for i in progress:
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                data, carry = collect(carry, rollout_policy)
            rollout_time = rollout_timer.last_time

            episode_stats.add(data)
            env_frames += data.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data))
            training_time = training_timer.last_time

            info.update(env.extra)
            info.update(env.stats_ema)  # step-wise exponential moving average of stats

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            if should_save(i):
                ckpt_path = save(policy, f"checkpoint_{i}")

            if aa.is_main_process():
                ScopedTimer.print_summary(clear=True)
                print(
                    OmegaConf.to_yaml(
                        {k: v for k, v in info.items() if isinstance(v, (float, int))}
                    )
                )
                print(f"Latest checkpoint: {ckpt_path}")
                run.log(info)

    if aa.is_main_process():
        ckpt_path = save(policy, "checkpoint_final")
        policy_eval = policy.get_rollout_policy("eval")
        info, trajs, stats = evaluate(
            env, policy_eval, render=cfg.eval_render, seed=cfg.seed
        )
        info["env_frames"] = env_frames
        run.log(info)
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    exit(0)


if __name__ == "__main__":
    main()
