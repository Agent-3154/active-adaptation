import hydra
import torch

from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass, field

from active_adaptation.envs.env_base import RewardGroup, mdp
from active_adaptation.pipeline_io import (
    RUN_STATE_FILENAME,
    get_run_state_dir,
    write_run_state,
)

from hydra.core.config_store import ConfigStore


defaults = [
    {"task": "A2/A2LocoManipSparse"},
]


@dataclass
class RelabelConfig:
    rollout_path: str
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name="relabel", node=RelabelConfig)


def mean_episode_return(
    reward: torch.Tensor,
    is_init: torch.Tensor,
    done: torch.Tensor,
) -> tuple[float, int]:
    """Mean undiscounted return over completed episodes in a stacked rollout.

    Accumulates ``reward[t]`` per env, resetting on ``is_init[t]``, and records
    the running sum when ``done[t]`` is true.
    """
    T, N = reward.shape[:2]
    ep_ret = torch.zeros(N, 1, device=reward.device, dtype=reward.dtype)
    completed: list[torch.Tensor] = []
    for t in range(T):
        ep_ret = ep_ret * (~is_init[t]).float()
        ep_ret = ep_ret + reward[t]
        if done[t].any():
            completed.append(ep_ret[done[t].squeeze(-1)].clone())
    if not completed:
        return float("nan"), 0
    returns = torch.cat(completed, dim=0)
    return returns.mean().item(), returns.numel()


def run(cfg: RelabelConfig) -> dict[str, str]:
    """Relabel rollout commands/rewards and return archive paths for downstream stages."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    command_cfg = dict(cfg.task.command)
    _target_ = command_cfg.pop("_target_")
    command = mdp.CommandV2.make(_target_, **command_cfg)

    rollout_path = Path(cfg.rollout_path).absolute()
    rollout = torch.load(rollout_path, weights_only=False)

    tensordict = rollout["stacked"]
    print(tensordict)

    T, N = tensordict.shape[:2]
    # rollout must contain "is_init" and ("next", "done")
    is_init = tensordict["is_init"]
    done = tensordict["next", "done"]
    assert is_init.shape == (T, N, 1), f"Expected `is_init` tensor with shape [T, N, 1], got {is_init.shape}"
    assert done.shape == (T, N, 1), f"Expected `(next, done)` tensor with shape [T, N, 1], got {done.shape}"

    print("Relabeling command...")
    command.relabel_command(tensordict)

    reward_cfg = cfg.task.reward
    for group_name, group_cfg in reward_cfg.items():
        key = ("next", "reward", group_name)
        if tensordict.get(key) is not None:
            continue
        reward_group = RewardGroup.create_from(group_name, group_cfg)
        if not reward_group.enabled:
            continue

        print(f"Relabeling reward group: {group_name}")
        rew = torch.zeros(T, N, 1, device=tensordict.device)
        for name, func in reward_group.funcs.items():
            print(f"\tRelabeling reward {name}...")
            rew = rew + func.weight * func.relabel(tensordict)
        tensordict[key] = rew
        mean_ret, n_episodes = mean_episode_return(rew, is_init, done)
        print(
            f"\tmean episode return ({group_name}): {mean_ret:.4f} "
            f"({n_episodes} completed episodes)"
        )

    rollout["stacked"] = tensordict
    save_path = rollout_path.with_suffix(".relabeled.pt")
    torch.save(rollout, save_path)
    print(f"Rollout saved to {save_path}")

    run_state = {
        "rollout_path": str(rollout_path),
        "relabeled_path": str(save_path.resolve()),
        "task": str(cfg.task.name),
    }
    run_state_dir = save_path.parent
    run_state_path = write_run_state(run_state, run_state_dir / RUN_STATE_FILENAME)
    print(f"Wrote run state to {run_state_path}")
    pipeline_dir = get_run_state_dir()
    if pipeline_dir is not None and pipeline_dir.resolve() != run_state_dir.resolve():
        write_run_state(run_state, pipeline_dir / RUN_STATE_FILENAME)
    return run_state


@hydra.main(config_path="../cfg", config_name="relabel", version_base=None)
def main(cfg: RelabelConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
