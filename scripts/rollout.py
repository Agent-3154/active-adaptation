"""
Roll out a policy and collect transitions for offline replay / inspection.

Writes a small ``torch.save`` payload (stacked steps, format version) under
``rollout/<task>/<algo>/<timestamp>/rollout.pt``.
"""

import datetime
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

import active_adaptation as aa

FILE_PATH = Path(__file__).parent


class RolloutWriter:
    """Append CPU transition rows and flush to ``rollout.pt`` in ``path``."""

    def __init__(self, path: Path, max_size: int = 2000):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._rows: list[TensorDict] = []

    def add(self, tensordict: TensorDict):
        assert tensordict.ndim == 1
        td = tensordict.detach().cpu()
        self._rows.append(td.clone())
        if len(self._rows) > self._max_size:
            self._rows = self._rows[-self._max_size :]
        return len(self._rows)

    def close(self) -> None:
        if not self._rows:
            return
        stacked: TensorDict = torch.stack(self._rows, dim=0)
        print(stacked)
        payload = {
            "format_version": 1,
            "writer_max_size": self._max_size,
            "stacked": stacked,
        }
        out_path = self.path / f"rollout_{stacked.shape[0]}_{stacked.shape[1]}.pt"
        torch.save(payload, out_path)
        size = out_path.stat().st_size
        if size >= 1024**3:
            human = f"{size / (1024**3):.2f} GiB"
        elif size >= 1024**2:
            human = f"{size / (1024**2):.2f} MiB"
        elif size >= 1024:
            human = f"{size / 1024:.2f} KiB"
        else:
            human = f"{size} B"
        print(f"Collected rollout disk usage: {size:,} bytes ({human}) at {out_path}")



@hydra.main(config_path="../cfg", config_name="rollout", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(cfg)
    obs_keys = list(env.observation_spec.keys())
    # whether to store ("next", ...) or not
    store_transitions = bool(cfg.store_transitions)
    exclude_keys = [("next", "stats"),]
    if not store_transitions:
        exclude_keys.extend(("next", key) for key in obs_keys)
    # wheter to run critic (if applicable)
    critic = bool(cfg.run_critic)
    rollout_policy = policy.get_rollout_policy("eval", critic=critic)

    env.eval()
    carry = env.reset()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer_path = FILE_PATH / "rollout" / f"{cfg.task.name}-{cfg.algo.name}" / timestamp
    writer = RolloutWriter(writer_path, max_size=cfg.num_steps)

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for _ in tqdm(range(cfg.num_steps)):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            td = td.exclude(*exclude_keys, inplace=True)
            writer.add(td)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
