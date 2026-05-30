"""
Roll out a policy and collect transitions for offline replay / inspection.

Writes a stacked transition archive and companion metadata JSON under
``rollout/<task>-<algo>/<timestamp>/``.
"""

import datetime

import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict

import active_adaptation as aa
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.rollout_io import (
    DEFAULT_ROLLOUT_ROOT,
    ROLLOUT_FORMAT_VERSION,
    episode_stats_to_metadata,
    format_bytes,
    save_rollout_with_metadata,
    update_metadata_shapes,
)

FILE_PATH = Path(__file__).parent


class RolloutWriter:
    """Append CPU transition rows and flush to disk in ``path``."""

    def __init__(self, path: Path, max_size: int = 2000, policy_name: str = ""):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._policy_name = policy_name
        self._rows: list[TensorDict] = []

    def add(self, tensordict: TensorDict):
        assert tensordict.ndim == 1
        td = tensordict.detach().cpu()
        self._rows.append(td.clone())
        if len(self._rows) > self._max_size:
            self._rows = self._rows[-self._max_size :]
        return len(self._rows)

    def close(
        self,
        *,
        episode_count: int = 0,
        episode_stats: dict[str, float] | None = None,
    ) -> None:
        if not self._rows:
            return
        stacked: TensorDict = torch.stack(self._rows, dim=0)
        print(stacked)
        payload = {
            "format_version": ROLLOUT_FORMAT_VERSION,
            "writer_max_size": self._max_size,
            "stacked": stacked,
        }
        out_path = self.path / f"rollout_{stacked.shape[0]}_{stacked.shape[1]}.pt"
        metadata = update_metadata_shapes(
            {
                "policy_name": self._policy_name,
                "episode_count": episode_count,
                "episode_stats": episode_stats or {},
            },
            stacked,
        )
        save_rollout_with_metadata(out_path, payload, metadata)
        size = out_path.stat().st_size
        print(
            f"Collected rollout disk usage: {size:,} bytes ({format_bytes(size)}) at {out_path}"
        )
        print(f"Episodes completed: {episode_count}")
        if episode_stats:
            for key, value in sorted(episode_stats.items()):
                print(f"  {key}: {value:.4f}")
        print(f"Wrote rollout metadata to {out_path.with_suffix('.json')}")


@hydra.main(config_path="../cfg", config_name="rollout", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(cfg)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    obs_keys = list(env.observation_spec.keys())
    store_transitions = bool(cfg.store_transitions)
    exclude_keys = [("next", "stats"),]
    if not store_transitions:
        exclude_keys.extend(("next", key) for key in obs_keys)
    critic = bool(cfg.run_critic)
    rollout_policy = policy.get_rollout_policy("eval", critic=critic)

    env.eval()
    carry = env.reset()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer_path = DEFAULT_ROLLOUT_ROOT / f"{cfg.task.name}-{cfg.algo.name}" / timestamp
    writer = RolloutWriter(
        writer_path,
        max_size=cfg.num_steps,
        policy_name=str(cfg.algo.name),
    )

    def is_private_key(key: str) -> bool:
        return isinstance(key, str) and key.startswith("_")

    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for _ in tqdm(range(cfg.num_steps)):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            episode_stats.add(td)

            private_keys = [key for key in td.keys(True, True) if is_private_key(key)]
            td = td.exclude(*private_keys, inplace=True)
            td = td.exclude(*exclude_keys, inplace=True)
            
            writer.add(td)

        episode_count = int(len(episode_stats))
        episode_stats_meta: dict[str, float] = {}
        if episode_count > 0:
            episode_stats_meta = episode_stats_to_metadata(episode_stats.pop())

    writer.close(episode_count=episode_count, episode_stats=episode_stats_meta)
    env.close()


if __name__ == "__main__":
    main()
