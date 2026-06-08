from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Randomization(MDPComponent, RegistryMixin):
    # set of fields that need to be expanded when using the mjlab backend
    mj_fields = tuple()

    def __init__(self, env):
        super().__init__(env)
        if self.env.backend == "mjlab":
            from active_adaptation.envs.backends.mjlab import MjlabSimAdapter

            sim: MjlabSimAdapter = self.env.sim
            fields = tuple(field for field in self.mj_fields if field not in sim._sim.expanded_fields)
            if fields:
                logging.info(f"[Mjlab Randomization] Expanding model fields: {fields}")
                sim._sim.expand_model_fields(fields)


class RandomizationV2(MDPComponent, RegistryMixin):
    """Environment-deferred domain randomization term.

    Like :class:`Randomization`, subclasses implement lifecycle hooks such as
    ``reset``, ``startup``, and ``update``.

    Unlike :class:`Randomization`, instances are constructed **without** an
    environment. Environment-bound state is created in :meth:`_initialize`,
    which the environment calls once at startup.
    """

    mj_fields = tuple()

    def __init__(self) -> None:
        self._initialized = False

    def _initialize(self, env: "_EnvBase") -> None:
        """Bind to ``env``. Called once at startup."""
        self.env = env
        if self.env.backend == "mjlab":
            from active_adaptation.envs.backends.mjlab import MjlabSimAdapter

            sim: MjlabSimAdapter = self.env.sim
            fields = tuple(field for field in self.mj_fields if field not in sim._sim.expanded_fields)
            if fields:
                logging.info(f"[Mjlab Randomization] Expanding model fields: {fields}")
                sim._sim.expand_model_fields(fields)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        """``True`` after :meth:`_initialize` has been called."""
        return self._initialized


__all__ = ["Randomization", "RandomizationV2"]
