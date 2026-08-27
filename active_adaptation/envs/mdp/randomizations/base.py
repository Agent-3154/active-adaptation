from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Randomization(MDPComponent, RegistryMixin):
    """Environment-deferred domain randomization term."""

    mj_fields = tuple()

    def __init__(self, env: "_EnvBase" | None = None) -> None:
        super().__init__()
        if env is not None:
            self._initialize(env)

    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        if self.env.backend == "mjlab":
            from active_adaptation.envs.backends.mjlab import MjlabSimAdapter

            sim: MjlabSimAdapter = self.env.sim
            fields = tuple(field for field in self.mj_fields if field not in sim._sim.expanded_fields)
            if fields:
                logging.info(f"[Mjlab Randomization] Expanding model fields: {fields}")
                sim._sim.expand_model_fields(fields)


__all__ = ["Randomization"]
