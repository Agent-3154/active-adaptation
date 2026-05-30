import torch
import torch.nn as nn


class RewardNormalizer(nn.Module):
    """Scale rewards by running discounted-return statistics."""

    def __init__(self, gamma: float, max_return: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.max_return = max_return
        self.eps = eps
        self.register_buffer("discounted_return", torch.zeros(1))
        self.register_buffer("discounted_return_abs_max", torch.zeros(1))
        self.register_buffer("return_mean", torch.zeros(1))
        self.register_buffer("return_var", torch.ones(1))
        self.register_buffer("return_count", torch.tensor(0.0))

    @torch.no_grad()
    def update(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        done: torch.Tensor,
    ):
        reset = torch.logical_or(terminated.bool(), done.bool()).float()
        reward = reward.detach()
        reset = reset.reshape_as(reward)

        if self.discounted_return.shape != reward.shape:
            self.discounted_return = torch.zeros_like(reward)
        self.discounted_return.mul_(self.gamma).mul_(1.0 - reset).add_(reward)
        self.discounted_return_abs_max.maximum_(
            self.discounted_return.detach().abs().max().reshape_as(
                self.discounted_return_abs_max
            )
        )

        samples = self.discounted_return.reshape(-1, 1)
        batch_mean = samples.mean(dim=0)
        batch_var = samples.var(dim=0, unbiased=False)
        batch_count = torch.as_tensor(
            samples.shape[0],
            dtype=self.return_count.dtype,
            device=self.return_count.device,
        )
        delta = batch_mean - self.return_mean
        total_count = self.return_count + batch_count
        ratio = batch_count / total_count.clamp_min(self.eps)
        new_mean = self.return_mean + delta * ratio
        m_a = self.return_var * self.return_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.return_count * ratio

        self.return_mean.copy_(new_mean)
        self.return_var.copy_(m2 / total_count.clamp_min(self.eps))
        self.return_count.copy_(total_count)

    def normalize(self, reward: torch.Tensor):
        var_denominator = torch.sqrt(self.return_var + self.eps)
        max_denominator = self.discounted_return_abs_max / self.max_return
        denominator = torch.maximum(var_denominator, max_denominator).clamp_min(
            self.eps
        )
        return reward / denominator.to(device=reward.device, dtype=reward.dtype)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        for key in (
            "discounted_return",
            "discounted_return_abs_max",
            "return_mean",
            "return_var",
            "return_count",
        ):
            if key in state_dict:
                setattr(self, key, state_dict[key].to(self.return_mean.device))
            elif strict:
                raise KeyError(f"Missing reward normalizer state: {key}")