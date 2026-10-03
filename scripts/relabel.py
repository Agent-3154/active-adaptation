import hydra
import torch

from omegaconf import OmegaConf, ListConfig
from pathlib import Path
from typing import List, Any, Optional, Sequence
from dataclasses import dataclass, field

from active_adaptation.envs.env_base import ObsGroup, RewardGroup, mdp
from active_adaptation.pipeline_io import (
    RUN_STATE_FILENAME,
    get_run_state_dir,
    write_run_state,
)
from active_adaptation.rollout_io import update_metadata_shapes, write_metadata_json

from hydra.core.config_store import ConfigStore


defaults = [
    {"task": "A2/A2LocoManipSparse"},
]


@dataclass
class RelabelConfig:
    rollout_path: str
    """Path to a stacked rollout archive (``.pt``)."""
    reward_groups: Optional[List[str]] = None
    """Reward groups to (re)label.

    * ``null`` (default): relabel only groups that are **absent** from the archive.
    * non-empty list: force-relabel these groups (overwrite if already present).
    """
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name="relabel", node=RelabelConfig)


def episode_returns(
    reward: torch.Tensor,
    is_init: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Undiscounted return of each completed episode. Shape ``[E, 1]``.

    ``reward``, ``is_init``, and ``done`` are ``[T, N, 1]``. Resets the running
    sum on ``is_init[t]`` before adding ``reward[t]``, and records it when
    ``done[t]`` is true.
    """
    time_steps, num_envs = is_init.shape[:2]
    reward = reward.reshape(time_steps, num_envs, 1)
    is_init = is_init.reshape(time_steps, num_envs, 1)
    done = done.reshape(time_steps, num_envs, 1)
    ep_ret = torch.zeros(num_envs, 1, device=reward.device, dtype=reward.dtype)
    completed: list[torch.Tensor] = []
    for t in range(time_steps):
        ep_ret = ep_ret * (~is_init[t]).float()
        ep_ret = ep_ret + reward[t]
        if done[t].any():
            completed.append(ep_ret[done[t].reshape(num_envs)].clone())
    if not completed:
        return reward.new_zeros(0, 1)
    return torch.cat(completed, dim=0)


def mean_episode_return(
    reward: torch.Tensor,
    is_init: torch.Tensor,
    done: torch.Tensor,
) -> tuple[float, int]:
    """Mean undiscounted return over completed episodes in a stacked rollout."""
    returns = episode_returns(reward, is_init, done)
    if returns.numel() == 0:
        return float("nan"), 0
    return returns.mean().item(), int(returns.shape[0])


def _episode_stats(
    tensordict,
    is_init: torch.Tensor,
    done: torch.Tensor,
    term_rewards: dict[str, torch.Tensor],
) -> tuple[dict[str, float], int]:
    """Mean episode length, per-term returns, and per-group returns."""
    stats: dict[str, float] = {}
    time_steps, num_envs = is_init.shape[:2]
    length = torch.ones(time_steps, num_envs, 1, device=is_init.device, dtype=torch.float32)
    length_ret = episode_returns(length, is_init, done)
    n_episodes = int(length_ret.shape[0])
    stats["stats/episode_len"] = (
        float("nan") if n_episodes == 0 else length_ret.mean().item()
    )
    for key, reward in term_rewards.items():
        mean, _ = mean_episode_return(reward, is_init, done)
        stats[key] = mean
    reward_td = tensordict.get(("next", "reward"))
    if reward_td is not None:
        for group_name, reward in reward_td.items():
            if not isinstance(reward, torch.Tensor):
                continue
            mean, _ = mean_episode_return(reward, is_init, done)
            stats[f"stats/{group_name}/return"] = mean
    return dict(sorted(stats.items())), n_episodes


def _make_component(base_cls, class_name: str, kwargs: dict):
    return base_cls.make(class_name, **kwargs)


def _where_done(keep, advance, done: torch.Tensor):
    """Keep ``keep`` where ``done`` is set, otherwise take ``advance``."""
    if torch.is_tensor(keep):
        mask = done
        while mask.ndim < keep.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, keep, advance)
    out = keep.clone()
    for key in keep.keys():
        out[key] = _where_done(keep[key], advance[key], done)
    return out


def _hold_across_done(value, done: torch.Tensor):
    """Value at the next step, held across ``done``. The last step stays."""
    if value.shape[0] <= 1:
        return value.clone()
    nxt = value.clone()
    nxt[:-1] = _where_done(value[:-1], value[1:], done[:-1])
    return nxt


def relabel_archive(
    rollout_path: Path | str,
    task_cfg,
    reward_groups: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Relabel ``rollout_path`` using ``task_cfg`` and write ``*.relabeled.pt``."""
    command_cfg = dict(task_cfg.command)
    _target_ = command_cfg.pop("_target_")
    command = mdp.Command.make(_target_, **command_cfg)

    rollout_path = Path(rollout_path).absolute()
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

    print("Observation groups: absent-only (skip keys already present)")
    for group_name, group_cfg in task_cfg.observation.items():
        if tensordict.get(group_name) is not None:
            print(f"Skipping observation group (already present): {group_name}")
            continue
        group = ObsGroup.create_from(
            group_name,
            group_cfg,
            make_component=_make_component,
            command=command,
        )
        print(f"Relabeling observation group: {group_name}")
        for name in group.funcs:
            print(f"\tRelabeling observation {name}...")
        obs = group.relabel(tensordict)
        tensordict[group_name] = obs
        tensordict["next", group_name] = _hold_across_done(obs, done)

    if isinstance(reward_groups, ListConfig):
        reward_groups = list(reward_groups)
    if reward_groups is None:
        print("Reward groups: absent-only (skip keys already present)")
    else:
        reward_groups = list(map(str, reward_groups))
        print(f"Reward groups: force-relabel {reward_groups}")
    
    def should_relabel_group(
        group_name: str,
    ) -> bool:
        """Decide whether to (re)label ``group_name``.

        * Specified list → only those names (overwrite).
        * ``None`` → only groups missing from ``tensordict``.
        """
        key = ("next", "reward", group_name)
        present = tensordict.get(key) is not None
        if reward_groups is None:
            return not present
        return group_name in reward_groups

    reward_cfg = task_cfg.reward
    if reward_groups is not None:
        unknown = [g for g in reward_groups if g not in reward_cfg]
        if unknown:
            raise KeyError(
                f"reward_groups not found in task.reward: {unknown}. "
                f"Available: {list(reward_cfg.keys())}"
            )

    term_stats: dict[str, torch.Tensor] = {}
    for group_name, group_cfg in reward_cfg.items():
        if not should_relabel_group(group_name):
            if reward_groups is not None and group_name not in reward_groups:
                print(f"Skipping reward group (not in reward_groups): {group_name}")
            elif tensordict.get(("next", "reward", group_name)) is not None:
                print(f"Skipping reward group (already present): {group_name}")
            continue

        reward_group = RewardGroup.create_from(
            group_name,
            group_cfg,
            make_component=_make_component,
        )
        if not reward_group.enabled:
            print(f"Skipping reward group (disabled): {group_name}")
            continue

        key = ("next", "reward", group_name)
        present = tensordict.get(key) is not None
        action = "Re-relabeling" if present else "Relabeling"
        print(f"{action} reward group: {group_name}")
        rew = torch.zeros(T, N, 1, device=tensordict.device)
        for name, func in reward_group.funcs.items():
            print(f"\tRelabeling reward {name}...")
            term = (func.weight * func.relabel(tensordict)).reshape(T, N, 1)
            rew = rew + term
            term_stats[f"stats/{group_name}/{name}"] = term
        tensordict[key] = rew

    episode_stats, n_episodes = _episode_stats(tensordict, is_init, done, term_stats)
    print(f"Relabeled episode stats ({n_episodes} completed episodes):")
    for key, value in episode_stats.items():
        print(f"  {key}: {value:.4f}")

    rollout["stacked"] = tensordict
    save_path = rollout_path.with_suffix(".relabeled.pt")
    torch.save(rollout, save_path)
    metadata = update_metadata_shapes(
        {
            "episode_count": n_episodes,
            "episode_stats": episode_stats,
            "task": str(task_cfg.name),
            "source_rollout": str(rollout_path),
        },
        tensordict,
    )
    write_metadata_json(metadata, save_path.with_suffix(".json"))
    print(f"Rollout saved to {save_path}")
    print(f"Wrote relabeled episode stats to {save_path.with_suffix('.json')}")

    return {
        "rollout_path": str(rollout_path),
        "relabeled_path": str(save_path.resolve()),
        "task": str(task_cfg.name),
    }


def run(cfg: RelabelConfig) -> dict[str, str]:
    """Relabel rollout commands/rewards and return archive paths for downstream stages."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    run_state = relabel_archive(cfg.rollout_path, cfg.task, cfg.reward_groups)
    save_path = Path(run_state["relabeled_path"])
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
