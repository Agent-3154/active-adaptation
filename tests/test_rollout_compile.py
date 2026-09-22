"""Behavior checks for compiled rollout regions (CPU or CUDA via env)."""
import os
import torch
from active_adaptation.utils.rollout_compile import (
    compile_rollout_function, finite_row_masks, sanitize_rows, rollout_compile_enabled,
)


def test_compiled_finite_masks_and_sanitize():
    device = os.environ.get('ROLLOUT_TEST_DEVICE', 'cpu')
    values = (torch.ones(7, 13, device=device), torch.ones(7, 3, 4, device=device))
    values[0][2, 0] = float('nan')
    values[1][4, 1, 2] = float('inf')
    mask, leaves = compile_rollout_function(finite_row_masks)(values)
    expected = torch.tensor([False, False, True, False, True, False, False], device=device)
    torch.testing.assert_close(mask, expected)
    assert leaves[0][2] and not leaves[0][4]
    assert leaves[1][4] and not leaves[1][2]
    cleaned = compile_rollout_function(sanitize_rows)(values, mask)
    for original, clean in zip(values, cleaned):
        torch.testing.assert_close(clean[~mask], original[~mask])
        assert torch.count_nonzero(clean[mask]) == 0


def test_compile_configuration_rejects_unknown_region():
    assert rollout_compile_enabled({'rollout_compile': ['observations']}, 'observations')
    for region in ('observations', 'rewards', 'nan_guard'):
        assert rollout_compile_enabled({}, region)
        assert not rollout_compile_enabled({'rollout_compile': []}, region)
    assert not rollout_compile_enabled({}, 'reset')
    assert not rollout_compile_enabled({'rollout_compile': ['observations']}, 'rewards')
    for unsupported in ('typo', 'reset'):
        try:
            rollout_compile_enabled({'rollout_compile': [unsupported]}, 'rewards')
        except ValueError:
            pass
        else:
            raise AssertionError(f'Unsupported compilation region accepted: {unsupported}')


def test_compiled_reward_values_preserve_ema_and_modifiers():
    from collections import OrderedDict
    from types import SimpleNamespace
    from active_adaptation.envs.env_base import RewardGroup
    from active_adaptation.envs.mdp.rewards.base import Reward
    device = os.environ.get('ROLLOUT_TEST_DEVICE', 'cpu')

    class SampleReward(Reward):
        def _compute(self):
            return self.values, self.active

    reward = SampleReward(weight=1.7, track_var=True)
    reward._initialize(SimpleNamespace(num_envs=4, device=device, command_manager=None))
    reward.values = torch.tensor([[1.], [float('nan')], [9.], [3.]], device=device)
    reward.active = torch.tensor([[True], [True], [False], [True]], device=device)
    reward._modifier = torch.tensor([[1.], [2.], [3.], [4.]], device=device)
    group = RewardGroup('test', OrderedDict(sample=reward))
    group.env = SimpleNamespace(num_envs=4, device=device)
    names = ('_modifier', '_ema_sum', '_ema_cnt', '_ema_sum_sq')
    before = {name: getattr(reward, name).clone() for name in names}
    expected = group.compute_values()
    expected_state = {name: getattr(reward, name).clone() for name in names}
    for name, value in before.items():
        setattr(reward, name, value)
    actual = compile_rollout_function(group.compute_values)()
    torch.testing.assert_close(actual, expected, equal_nan=True)
    for name in names:
        torch.testing.assert_close(getattr(reward, name), expected_state[name])


def test_quat_from_matrix_keeps_fixed_shape_in_compiled_observation():
    from active_adaptation.utils.math import matrix_from_quat, quat_from_matrix

    device = os.environ.get('ROLLOUT_TEST_DEVICE', 'cpu')
    generator = torch.Generator().manual_seed(20260922)
    quats = torch.randn(128, 11, 4, generator=generator).to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    matrices = matrix_from_quat(quats)

    recovered = compile_rollout_function(quat_from_matrix)(matrices)

    assert recovered.shape == quats.shape
    # q and -q represent the same rotation.
    agreement = (recovered * quats).sum(dim=-1).abs()
    torch.testing.assert_close(agreement, torch.ones_like(agreement), atol=1e-5, rtol=0)
