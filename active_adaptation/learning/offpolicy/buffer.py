from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from typing import Any, Dict, Optional, Tuple, Union, Callable
from pathlib import Path

from torchrl.data.replay_buffers.samplers import PrioritizedSampler

# Written by :mod:`active_adaptation.scripts.rollout` and read by :meth:`ReplayBuffer.from_rollout`.
ROLLOUT_ARCHIVE_NAME = "rollout.pt"
ROLLOUT_FORMAT_VERSION = 1


class ReplayBuffer:
    """Ring replay storage (no TorchRL :class:`~torchrl.data.ReplayBuffer`).

    Pass the ring storage :class:`~tensordict.TensorDict` (batch ``[max_size, num_envs]``)
    to ``__init__``, normally built by :meth:`from_fake` or :meth:`from_rollout`.
    """

    def __init__(
        self,
        buffer_tensordict: TensorDict,
        *,
        current_size: int = 0,
        write_ptr: int = 0,
        per_alpha: Optional[float] = None,
        per_beta: float = 1.0,
        per_eps: float = 1e-8,
        per_generator: Optional[torch.Generator] = None,
    ):
        if len(buffer_tensordict.batch_size) != 2:
            raise ValueError(
                f"buffer_tensordict must have batch rank 2 [max_size, num_envs], got {buffer_tensordict.batch_size}."
            )
        self._td = buffer_tensordict
        self.max_size = int(self._td.shape[0])
        self.num_envs = int(self._td.shape[1])
        self.device = self._td.device
        self._current_size = current_size
        self._ptr = write_ptr

        self._per: Optional[PrioritizedSampler] = None
        self._init_prioritized_sampler(
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_eps=per_eps,
            per_generator=per_generator,
        )

    def __repr__(self) -> str:
        keys = list(self.keys())
        if len(keys) <= 12:
            key_desc = "[" + ", ".join(repr(k) for k in keys) + "]"
        else:
            key_desc = (
                "["
                + ", ".join(repr(k) for k in keys[:12])
                + f", … (+{len(keys) - 12} keys)]"
            )
        if self._per is None:
            sampling = "uniform"
        else:
            alpha = getattr(self._per, "alpha", None)
            beta = getattr(self._per, "beta", None)
            sampling = f"PER(α={alpha}, β={beta})"
        return (
            f"{self.__class__.__name__}("
            f"ring={len(self)}/{self.max_size}×{self.num_envs}, "
            f"write_ptr={self._ptr}, {sampling}, device={self.device}, "
            f"keys={key_desc})"
        )
    
    def keys(self):
        return self._td.keys(True, True)

    def _init_prioritized_sampler(
        self,
        *,
        per_alpha: Optional[float],
        per_beta: float,
        per_eps: float,
        per_generator: Optional[torch.Generator],
    ) -> None:
        if per_alpha is None:
            return
        cap = self.max_size * self.num_envs
        self._per = PrioritizedSampler(
            max_capacity=cap,
            alpha=per_alpha,
            beta=per_beta,
            eps=per_eps,
            dtype=torch.float,
        )
        self._per._rng = per_generator
    
    def select_(self, *keys: str) -> ReplayBuffer:
        self._td = self._td.select(*keys, inplace=True, strict=True)
        return self
    
    def exclude_(self, *keys: str) -> ReplayBuffer:
        self._td = self._td.exclude(*keys, inplace=True, strict=True)
        return self

    @classmethod
    def from_fake(
        cls,
        max_size: int,
        fake_tensordict: TensorDict,
        *,
        per_alpha: Optional[float] = None,
        per_beta: float = 1.0,
        per_eps: float = 1e-8,
        per_generator: Optional[torch.Generator] = None,
    ) -> ReplayBuffer:
        """Build ring storage ``[max_size, num_envs]`` from a one-step template and construct the buffer."""
        td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        return cls(
            td,
            current_size=0,
            write_ptr=0,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_eps=per_eps,
            per_generator=per_generator,
        )

    @classmethod
    def from_rollout(
        cls,
        path: Union[str, Path],
        *,
        max_size: Optional[int] = None,
        per_alpha: Optional[float] = None,
        per_beta: float = 1.0,
        per_eps: float = 1e-8,
        per_generator: Optional[torch.Generator] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> ReplayBuffer:
        """Load from a rollout archive produced by :mod:`active_adaptation.scripts.rollout`.

        The archive is a ``torch.save`` dict with keys ``format_version``, ``stacked``
        (TensorDict with batch ``[T, num_envs]``), and optionally ``writer_max_size``.

        Args:
            path: File ``rollout.pt`` or directory containing it.
            max_size: Ring capacity. Defaults to ``max(writer_max_size, T)`` from the archive.
        """
        root = Path(path)
        file = root if root.suffix == ".pt" else root / ROLLOUT_ARCHIVE_NAME
        if not file.is_file():
            raise FileNotFoundError(f"No rollout archive at {file}")

        payload: Dict[str, Any] = torch.load(file, map_location=map_location, weights_only=False)
        version = payload.get("format_version")
        if version != ROLLOUT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported rollout format_version={version!r}; expected {ROLLOUT_FORMAT_VERSION}."
            )
        stacked: TensorDict = payload["stacked"]
        if len(stacked.batch_size) < 2:
            raise ValueError(
                f"Expected stacked transitions with batch [T, num_envs], got batch_size={stacked.batch_size}."
            )
        T = int(stacked.batch_size[0])
        if T < 1:
            raise ValueError("Rollout archive contains zero transitions.")

        writer_max = int(payload.get("writer_max_size", T))
        ring_cap = max_size if max_size is not None else max(writer_max, T)

        take = min(T, ring_cap)
        if T >= ring_cap:
            td = stacked[-ring_cap:].clone()
        else:
            row0 = stacked[0]
            td = row0.expand(ring_cap, *row0.shape).clone()
            td[:take] = stacked[:take]
        ptr = take % ring_cap

        out = cls(
            td,
            current_size=take,
            write_ptr=ptr,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_eps=per_eps,
            per_generator=per_generator,
        )
        if out._per is not None:
            for wrow in range(take):
                flat = (
                    torch.arange(out.num_envs, dtype=torch.long, device=torch.device("cpu"))
                    + int(wrow) * out.num_envs
                )
                out._per.mark_update(flat)
        return out

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
        idx_long = idx_flat.to(dtype=torch.long)
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

        return (
            index.to(device=self._td.device, dtype=torch.long),
            importance.to(device=self._td.device, dtype=torch.float32),
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
            weight = torch.ones(
                batch_size, device=self._td.device, dtype=torch.float32
            )

        t, e = torch.unravel_index(idx_flat, (len(self), self._td.shape[1]))
        if steps == 1:
            samples = self._td[t, e]#.rename("env")
        else:
            ts = (t.unsqueeze(0) + torch.arange(steps, device=self._td.device).unsqueeze(1)) % len(self)
            samples = self._td[ts, e]#.rename("time", "env")
            assert samples.shape[:2] == (steps, batch_size)
        samples = self._annotate_sampling_meta(samples, idx_flat, steps, weight)
        return samples

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

    def compute_return(
        self,
        reward_key: str | Tuple[str, ...],
        gamma: float,
        fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        done_key: str | Tuple[str, ...] = ("next", "done"),
        discount_key: str | Tuple[str, ...] = ("next", "discount"),
        is_init_key: str = "is_init",
    ) -> None:
        """Compute Monte Carlo discounted returns and store them under ``ret``.

        Also stores ``ret_valid`` (bool, shape ``[max_size, N, 1]``): a step's
        return is valid only if its episode terminates (``done``) within the
        filled buffer rows ``[:T]``. Truncated episodes that reach the buffer
        end without ``done`` get ``ret_valid=False`` for all their steps.

        Args:
            reward_key: Buffer key for rewards, e.g. ``("next", "reward")``.
            gamma: Discount factor.
            fn: Maps raw reward tensor to scalar reward, e.g.
                ``lambda x: x.sum(-1, keepdim=True).clamp_min(0.)``.
            done_key: Key marking episode termination after the transition.
            discount_key: Per-step discount multiplier with shape ``[T, N, 1]``
                (defaults to ones if absent).
            is_init_key: Key marking the first step of an episode (shape ``[T, N, 1]``).
        
        TODO: handle bootstraping. the current return is incomplete.
        """
        if "ret" in self._td.keys(True, True) or "ret_valid" in self._td.keys(True, True):
            raise ValueError("Return keys already exist in buffer.")

        T, N = len(self), self.num_envs
        if T == 0:
            raise RuntimeError("Cannot compute returns on an empty ReplayBuffer.")

        rew = fn(self._td.get(reward_key)[:T]).float()
        assert rew.ndim == 3, f"Expected reward tensor with shape [T, N, *], got {rew.shape}"

        done = self._td.get(done_key)[:T]
        assert done.shape == (T, N, 1), f"Expected done tensor with shape [T, N, 1], got {done.shape}"

        discount = self._td.get(discount_key, default=None)
        if discount is None:
            discount = torch.ones(T, N, 1, device=rew.device, dtype=rew.dtype)
        else:
            discount = discount[:T].float()
            assert discount.shape == (T, N, 1), (
                f"Expected discount tensor with shape [T, N, 1], got {discount.shape}"
            )

        is_init = self._td.get(is_init_key)[:T].bool()
        assert is_init.shape == (T, N, 1), (
            f"Expected is_init tensor with shape [T, N, 1], got {is_init.shape}"
        )

        ret = torch.zeros_like(rew)
        ret_valid = torch.zeros(T, N, 1, dtype=torch.bool, device=rew.device)
        running = torch.zeros(N, rew.shape[-1], device=rew.device, dtype=rew.dtype)
        running_valid = torch.zeros(N, 1, dtype=torch.bool, device=rew.device)
        nonterminal = 1.0 - done.float()

        for t in reversed(range(T)):
            running = rew[t] + gamma * discount[t] * nonterminal[t] * running
            ret[t] = running
            running = running * (~is_init[t]).float()

            # valid turns true when seeing an episode end
            running_valid = running_valid | done[t]
            ret_valid[t] = running_valid

        self._td.set("ret", ret)
        self._td.set("ret_valid", ret_valid)

