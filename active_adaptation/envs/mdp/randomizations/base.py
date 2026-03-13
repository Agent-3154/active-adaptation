from active_adaptation.registry import RegistryMixin

from ..base import MDPComponent


class Randomization(MDPComponent, RegistryMixin):
    def __init__(self, env):
        super().__init__(env)


__all__ = ["Randomization"]
