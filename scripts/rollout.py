"""
This script is used to rollout a policy and collect data.
"""

import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

import active_adaptation as aa


aa.import_algorithms()
FILE_PATH = Path(__file__).parent


class RolloutWriter:
    def __init__(self, path: Path, max_size: int = 2000):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        storage = LazyMemmapStorage(max_size=max_size, scratch_dir=path / "storage")
        self.rb = ReplayBuffer(storage=storage)

    def add(self, tensordict: TensorDict):
        assert tensordict.ndim == 1
        tensordict = tensordict.cpu(non_blocking=True)
        self.rb.add(tensordict)
        return len(self.rb)
    
    def close(self):
        self.rb.dumps(self.path)


@hydra.main(config_path="../cfg", config_name="rollout", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg)

    from active_adaptation.helpers import make_env_policy
    env, policy = make_env_policy(cfg)
    rollout_policy = policy.get_rollout_policy("eval")

    env.eval()
    carry = env.reset()

    writer_path = FILE_PATH / "rollout" / cfg.task.name / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    writer = RolloutWriter(writer_path, max_size=cfg.num_steps)

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):

        for _ in tqdm(range(cfg.num_steps)):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            writer.add(td)
    
    writer.close()
    env.close()


if __name__ == "__main__":
    main()