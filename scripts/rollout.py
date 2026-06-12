"""
Roll out a policy and collect transitions for offline replay / inspection.

Writes a stacked transition archive and companion metadata JSON under
``rollout/<task>-<algo>/<timestamp>/``.
"""

import datetime

import hydra
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict, NonTensorData

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


DEFAULTS = [
    {"task": "Velocity"},
    {"algo": "ppo"},
    "_self_",
]


@dataclass
class IsaacAppConfig:
    """Isaac Lab AppLauncher settings (resolved from parent config)."""

    headless: bool = "${..headless}"
    """Mirror ``headless``; passed to Isaac Lab's AppLauncher."""
    enable_cameras: bool = False
    """Keep cameras off during headless rollout collection."""


@dataclass
class RolloutConfig:
    """Hydra root config for policy rollout and transition collection."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    """Hydra defaults list: task config, algo config, then this config."""
    hydra: HydraConf = field(default_factory=HydraConf)
    """Hydra runtime settings (output directory, etc.)."""
    num_steps: Any = "${oc.select:task.max_episode_length,1000}"
    """Number of env steps to collect; defaults to ``task.max_episode_length``."""
    headless: bool = True
    """Run simulation without a rendering window."""
    backend: str = "isaac"
    """Simulation backend: ``isaac``, ``mujoco``, ``mjlab``, or ``motrix``."""
    device: str = "cuda"
    """Torch device for policy inference (e.g. ``cuda``, ``cpu``)."""
    seed: int = 42
    """Random seed (offset by local rank in distributed runs)."""
    store_transitions: bool = True
    """Keep full next-step observations in saved transitions."""
    run_critic: bool = True
    """Run the critic during rollout (adds value estimates to the policy path)."""
    checkpoint_path: Optional[str] = None
    """Path or WandB URI to a policy checkpoint; ``null`` starts from scratch."""
    discard_unused_obs: bool = False
    """Drop observation groups not listed in ``algo.in_keys``."""
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    """Backend-specific application launcher config."""


cs = ConfigStore.instance()
cs.store(
    name="rollout",
    node=RolloutConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_rollout/${now:%Y-%m-%d}/${now:%H-%M-%S}-${task.name}-${algo.name}"
            )
        )
    ),
)


FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


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
        td = tensordict.detach().clone().cpu(non_blocking=True)
        self._rows.append(td)
        if len(self._rows) > self._max_size:
            self._rows = self._rows[-self._max_size :]
        return len(self._rows)

    def close(
        self,
        env_meta: dict[str, float],
        *,
        episode_count: int = 0,
        episode_stats: dict[str, float] | None = None,
    ) -> None:
        if not self._rows:
            return
        stacked: TensorDict = torch.stack(self._rows, dim=0)
        stacked["env_meta"] = NonTensorData(env_meta)
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
        if episode_stats:
            for key, value in sorted(episode_stats.items()):
                print(f"  {key}: {value:.4f}")
        print(
            f"Collected rollout disk usage: {size:,} bytes ({format_bytes(size)}) at {out_path}"
        )
        print(f"Episodes completed: {episode_count}")
        print(f"Wrote rollout metadata to {out_path.with_suffix('.json')}")


@hydra.main(config_path=str(CONFIG_PATH), config_name="rollout", version_base=None)
def main(cfg: RolloutConfig):
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
            command_state = env.command_manager.get_state()
            episode_stats.add(td)

            private_keys = [key for key in td.keys(True, True) if is_private_key(key)]
            td = td.exclude(*private_keys, inplace=True)
            td = td.exclude(*exclude_keys, inplace=True)
            td["command_state"] = command_state
            writer.add(td)

        episode_count = int(len(episode_stats))
        episode_stats_meta: dict[str, float] = {}
        if episode_count > 0:
            episode_stats_meta = episode_stats_to_metadata(episode_stats.pop())

    writer.close(
        env_meta = {"step_dt": env.step_dt, "physics_dt": env.physics_dt},
        episode_count=episode_count,
        episode_stats=episode_stats_meta
    )
    env.close()


if __name__ == "__main__":
    main()
