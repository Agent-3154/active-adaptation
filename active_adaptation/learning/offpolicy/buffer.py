import numpy as np
import torch
from tensordict import TensorDict
from typing import Optional, Tuple, Union

from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        fake_tensordict: TensorDict,
        *,
        per_alpha: Optional[float] = None,
        per_beta: float = 1.0,
        per_eps: float = 1e-8,
        per_dtype: torch.dtype = torch.float32,
        per_generator: Optional[torch.Generator] = None,
    ):
        self.max_size = max_size
        self.num_envs = fake_tensordict.shape[0]
        self.device = fake_tensordict.device
        self._current_size = 0
        self._td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        self._ptr = 0

        self._per: Optional[PrioritizedSampler] = None
        if per_alpha is not None:
            cap = max_size * self.num_envs
            dtype = per_dtype
            if dtype == torch.float32:
                dtype = torch.float
            elif dtype == torch.float64:
                dtype = torch.double
            self._per = PrioritizedSampler(
                max_capacity=cap,
                alpha=per_alpha,
                beta=per_beta,
                eps=per_eps,
                dtype=dtype,
            )
            self._per._rng = per_generator

    @property
    def prioritized(self):
        return self._per is not None

    def flat_index(self, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Map ring indices to the flat layout used by :meth:`sample` and :meth:`update_priority`."""
        return t * self._td.shape[1] + e

    def update_priority(
        self,
        flat_index: Union[torch.Tensor, int],
        priority: Union[torch.Tensor, float],
    ) -> None:
        """Update PER priorities (e.g. :math:`|\\delta|`). Uses Schaul-style :math:`(p+\\varepsilon)^\\alpha` internally.

        ``flat_index`` matches the flattened layout ``t * num_envs + env`` consistent with :meth:`sample`.
        """
        if self._per is None:
            raise RuntimeError("Prioritized replay is disabled (per_alpha=None).")

        idx = torch.as_tensor(flat_index, dtype=torch.long, device=torch.device("cpu")).reshape(-1)
        pr = torch.as_tensor(priority, dtype=torch.float, device=torch.device("cpu")).reshape(-1)
        if pr.numel() == 1 and idx.numel() > 1:
            pr = pr.expand_as(idx)
        self._per.update_priority(idx, pr)

    def _annotate_sampling_meta(
        self,
        samples: TensorDict,
        idx_flat: torch.Tensor,
        steps: int,
        priority_weight: torch.Tensor,
    ) -> TensorDict:
        """Attach ``replay_flat_index`` (always) and ``priority_weight`` (PER or all-ones).

        Segment starts use the flattened layout ``t * num_envs + env``."""
        priority_weight_batched = (
            priority_weight
            if steps == 1
            else priority_weight.view(1, -1).expand(steps, -1).contiguous()
        )
        idx_long = idx_flat.long()
        rfi = (
            idx_long
            if steps == 1
            else idx_long.view(1, -1).expand(steps, -1).contiguous()
        )
        return samples.set("priority_weight", priority_weight_batched).set("replay_flat_index", rfi)

    def push(self, tensordict: TensorDict):
        wrow = self._ptr
        self._td[wrow] = tensordict.to(self.device)
        self._ptr = (self._ptr + 1) % self._td.shape[0]
        self._current_size = min(self._current_size + 1, self.max_size)

        if self._per is not None:
            flat = torch.arange(self.num_envs, dtype=torch.long) + int(wrow) * self.num_envs
            self._per.mark_update(flat)

    append = push

    def last(self, steps: int) -> TensorDict:
        """
        Returns the last `steps` samples from the buffer.
        """
        assert len(self) > steps, "Not enough samples in buffer"
        if self._ptr > steps:
            samples = self._td[self._ptr - steps : self._ptr].clone()
        else:
            part1 = self._td[-(steps - self._ptr) :]
            part2 = self._td[: self._ptr]
            samples = torch.cat([part1, part2], dim=0)
        assert samples.shape[0] == steps, "Not enough samples in buffer"
        return samples

    @property
    def num_samples(self):
        return self._td.shape[1] * len(self)

    def _sample_prioritized_flat(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ps = self._per
        n = self.num_samples
        p_sum = ps._sum_tree.query(0, n)
        p_min = ps._min_tree.query(0, n)
        if p_sum <= 0 or p_min <= 0:
            raise RuntimeError("non-positive prioritized mass; check replay buffer PRI setup.")

        if ps._rng is None:
            mass = np.random.uniform(0.0, p_sum, size=batch_size)
        else:
            mass = torch.rand(batch_size, generator=ps._rng) * p_sum

        index = torch.as_tensor(ps._sum_tree.scan_lower_bound(mass))
        if not index.ndim:
            index = index.unsqueeze(0)
        index.clamp_max_(n - 1)

        weight = torch.as_tensor(ps._sum_tree[index])
        zero_weight = weight == 0
        while zero_weight.any():
            index = torch.where(zero_weight, index - 1, index)
            if (index < 0).any():
                raise RuntimeError("Prioritized replay sampling failed to find suitable indices.")
            weight = torch.as_tensor(ps._sum_tree[index])
            zero_weight = weight == 0

        importance = torch.pow(weight / p_min, -ps.beta)

        wdtype = (
            self._td.dtype
            if self._td.dtype.is_floating_point
            else torch.float32
        )
        return (
            index.to(self._td.device),
            importance.to(device=self._td.device, dtype=wdtype),
        )

    def sample(self, batch_size: int, steps: int = 1) -> TensorDict:
        """Draw a batch (optionally n-step segments along ring time per env).

        Every batch includes ``replay_flat_index`` (flattened ``t * num_envs + env`` for
        segment starts). ``priority_weight`` is the PER importance sampling weight when
        ``per_alpha`` is set; otherwise all ones (same tensor layout so learners need not branch on keys).

        Call :meth:`update_priority` with ``replay_flat_index`` only when
        ``prioritized`` is true.
        """
        if len(self) == 0 or self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        if self._per is not None:
            idx_flat, weight = self._sample_prioritized_flat(batch_size)
        else:
            idx_flat = torch.randint(
                0, self.num_samples, (batch_size,), device=self._td.device
            )
            weight = torch.ones(batch_size, device=self._td.device)

        t, e = torch.unravel_index(idx_flat, (len(self), self._td.shape[1]))
        if steps == 1:
            samples = self._td[t, e]
        else:
            ts = (t.unsqueeze(0) + torch.arange(steps, device=self._td.device).unsqueeze(1)) % len(self)
            samples = self._td[ts, e]
            assert samples.shape[:2] == (steps, batch_size)

        return self._annotate_sampling_meta(samples, idx_flat, steps, weight)

    def sample_sequential(
        self,
        batch_size: int,
        steps: int = 1,
        last_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sequential_prob: float = 0.0,
        sequential_offset: int = -1,
    ) -> Tuple[TensorDict, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample transitions with optional temporal correlation along the replay ring.

        Most indices are drawn i.i.d., like :meth:`sample`. With probability
        ``sequential_prob``, each element instead steps **backward** along the ring
        time index for the same env id, matching the direction rewards propagate
        under dynamic programming (earlier stored step = one step back in ring).

        Ring indices wrap with ``% len(self)``. After a minibatch, pass the returned
        ``(t, e)`` back as ``last_indices`` on the next call to chain segments.

        Args:
            batch_size: Number of transitions (or sequences when ``steps > 1``).
            steps: If ``> 1``, return length-``steps`` segments starting at each
                chosen ``t`` (forward along the ring from that start).
            last_indices: Previous ``(t, e)`` tensors from this method, same length
                as ``batch_size``, or ``None`` for independent draws only.
            sequential_prob: In ``(0, 1]``, fraction of elements (in expectation)
                that reuse ``last_indices`` shifted by ``sequential_offset`` instead
                of a fresh random index. ``0`` disables chaining.
            sequential_offset: Added to ``last_indices[0]`` before wrapping; default
                ``-1`` is one step **backward** in ring time (reward backup direction).

        Returns:
            ``(samples, (t, e))`` — batch/sequence tensordict and the time/env indices
            used for each row (for feeding back as ``last_indices``).
        """
        if len(self) == 0 or self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        indices_flat = torch.randint(
            0, self.num_samples, (batch_size,), device=self._td.device
        )
        t, e = torch.unravel_index(indices_flat, (len(self), self._td.shape[1]))

        if last_indices is not None and sequential_prob > 0.0:
            use_new = torch.rand(batch_size, device=self._td.device) > sequential_prob
            t = torch.where(use_new, t, (last_indices[0] + sequential_offset) % len(self))
            e = torch.where(use_new, e, last_indices[1])

        if steps == 1:
            samples = self._td[t, e].squeeze(0)
        else:
            ts = (
                t.unsqueeze(0)
                + torch.arange(steps, device=self._td.device).unsqueeze(1)
            ) % len(self)
            samples = self._td[ts, e]
            assert samples.shape[:2] == (steps, batch_size)

        return samples, (t, e)

    def __len__(self):
        return self._current_size
