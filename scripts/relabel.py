import hydra
import torch

from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass, field

import active_adaptation as aa
from active_adaptation.envs.env_base import RewardGroup, mdp

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


@hydra.main(config_path="../cfg", config_name="relabel", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # This script does not instantiate an environment


    command_cfg = dict(cfg.task.command)
    _target_ = command_cfg.pop("_target_")
    command = mdp.CommandV2.make(_target_, **command_cfg)

    rollout_path = Path(cfg.rollout_path).absolute()
    rollout = torch.load(rollout_path, weights_only=False)
    
    tensordict = rollout["stacked"]
    print(tensordict)

    T, N = tensordict.shape[:2]
    print(f"Relabeling command...")
    tensordict = command.relabel_command(tensordict)

    reward_cfg = cfg.task.reward
    for group_name, group_cfg in reward_cfg.items():
        key = ("next", "reward", group_name)
        if tensordict.get(key) is not None:
            continue
        print(f"Relabeling reward group: {group_name}")
        reward_group = RewardGroup.create_from(group_name, group_cfg)
        rew = torch.zeros(T, N, 1, device=tensordict.device)
        for name, func in reward_group.funcs.items():
            print(f"\tRelabeling reward {name}...")
            rew = rew + func.weight * func.relabel(tensordict)
        tensordict[key] = rew
    
    rollout["stacked"] = tensordict
    save_path = rollout_path.with_suffix(".relabeled.pt")
    torch.save(rollout, save_path)
    print(f"Rollout saved to {save_path}")


if __name__ == "__main__":
    main()
