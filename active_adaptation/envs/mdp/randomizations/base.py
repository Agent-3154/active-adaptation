import logging
from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


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


__all__ = ["Randomization"]
