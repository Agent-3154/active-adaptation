import torch

from active_adaptation.envs.mdp.rewards.base import Reward


def test_reward_ema_ignores_nonfinite_and_inactive_values():
    reward = object.__new__(Reward)
    reward._ema_sum = torch.zeros(1)
    reward._ema_cnt = torch.zeros(1)
    reward._ema_sum_sq = torch.zeros(1)

    reward._update_ema(
        torch.tensor([[1.0], [float("nan")], [100.0], [3.0]]),
        torch.tensor([[True], [True], [False], [True]]),
    )

    mean, var = reward.get_ema_stats()
    assert mean.item() == 2.0
    assert var is not None and var.item() == 1.0
