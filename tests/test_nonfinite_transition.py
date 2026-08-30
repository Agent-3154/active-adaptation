import torch
from tensordict import TensorDict

from active_adaptation.envs.env_base import _sanitize_nonfinite_env_rows


def test_nonfinite_transition_rows_are_zeroed_and_terminated():
    td = TensorDict(
        {
            "reward": torch.tensor([[1.0], [float("nan")], [3.0]]),
            "policy": torch.tensor([[1.0, 2.0], [4.0, 5.0], [float("inf"), 7.0]]),
            "stats": TensorDict(
                {"return": torch.tensor([[8.0], [9.0], [10.0]])},
                batch_size=[3],
            ),
            "terminated": torch.zeros(3, 1, dtype=torch.bool),
            "truncated": torch.ones(3, 1, dtype=torch.bool),
            "done": torch.zeros(3, 1, dtype=torch.bool),
            "discount": torch.ones(3, 1),
        },
        batch_size=[3],
    )

    invalid = _sanitize_nonfinite_env_rows(td)

    assert invalid.tolist() == [False, True, True]
    assert td["reward"].tolist() == [[1.0], [0.0], [0.0]]
    assert td["policy"].tolist() == [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]
    assert td["stats", "return"].tolist() == [[8.0], [0.0], [0.0]]
    assert td["terminated"].squeeze(-1).tolist() == [False, True, True]
    assert td["truncated"].squeeze(-1).tolist() == [True, False, False]
    assert td["done"].squeeze(-1).tolist() == [False, True, True]
    assert td["discount"].squeeze(-1).tolist() == [1.0, 0.0, 0.0]
