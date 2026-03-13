import torch

from active_adaptation.envs.mdp.terminations.base import Termination

from .command import Game


class both_terminate(Termination[Game]):
    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        termination = termination.reshape(-1, 2)
        termination = termination | termination.flip(1)
        return termination.reshape(self.num_envs, 1)


class caught_termination(Termination[Game]):
    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return self.command_manager.target_caught_time > 0.1


__all__ = ["both_terminate", "caught_termination"]
