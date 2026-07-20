from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Callable
from pathlib import Path

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
        observation_keys: Sequence[str],
        *,
        current_size: int = 0,
        write_ptr: int = 0,
        fake_bootstrap: bool = False,
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

        self.fake_bootstrap = fake_bootstrap
        self.observation_keys = tuple(observation_keys)
        if self.fake_bootstrap:
            self._td["next"] = self._td["next"].exclude(*self.observation_keys)

        self.mem_bytes = self.estimate_memory_nbytes()

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
        mem = format_nbytes(self.mem_bytes)
        return (
            f"{self.__class__.__name__}("
            f"ring={len(self)}/{self.max_size}×{self.num_envs}, "
            f"write_ptr={self._ptr}, sampling=uniform, mem≈{mem}, device={self.device}, "
            f"keys={key_desc})"
        )

    def estimate_memory_nbytes(self) -> int:
        """Approximate bytes held by tensor entries in replay storage."""
        nbytes = 0
        for _, value in self._td.items(True, True):
            if torch.is_tensor(value):
                nbytes += value.numel() * value.element_size()
        return nbytes
    
    def keys(self):
        return self._td.keys(True, True)
    
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
        observation_keys: Sequence[str],
        fake_bootstrap: bool = False,
    ) -> ReplayBuffer:
        """Build ring storage ``[max_size, num_envs]`` from a one-step template and construct the buffer."""
        td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        return cls(
            td,
            observation_keys,
            current_size=0,
            write_ptr=0,
            fake_bootstrap=fake_bootstrap,
        )

    @classmethod
    def from_rollout(
        cls,
        path: Union[str, Path],
        *,
        observation_keys: Sequence[str],
        max_size: Optional[int] = None,
        map_location: Union[str, torch.device] = "cpu",
        fake_bootstrap: bool = False,
    ) -> ReplayBuffer:
        """Load from a rollout archive produced by :mod:`active_adaptation.scripts.rollout`.

        The archive is a ``torch.save`` dict with keys ``format_version``, ``stacked``
        (TensorDict with batch ``[T, num_envs]``), and optionally ``writer_max_size``.

        Args:
            path: File ``rollout.pt`` or directory containing it.
            observation_keys: Keys treated as observations (required for bootstrap / term_obs).
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

        return cls(
            td,
            observation_keys,
            current_size=take,
            write_ptr=ptr,
            fake_bootstrap=fake_bootstrap,
        )

    def push(self, tensordict: TensorDict):
        wrow = self._ptr
        self._td[wrow] = tensordict.to(self.device)
        self._ptr = (self._ptr + 1) % self._td.shape[0]
        self._current_size = min(self._current_size + 1, self.max_size)

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

    def sample(self, batch_size: int, steps: int = 1, next_obs: bool = False, term_obs: bool = False) -> TensorDict:
        """Draw a batch (optionally n-step segments along ring time per env).

        Args:
            batch_size: Number of independent starts (env indices).
            steps: Segment length along ring time. ``1`` returns a flat batch.
            next_obs: If ``fake_bootstrap`` is set, reconstruct ``next`` observations
                from the following ring row (repeat current obs when ``done``).
            term_obs: Attach the observation at episode end under ``term_obs``.
                Requires :meth:`compute_return` (uses ``steps_to_go``). Only
                supported for ``steps == 1`` on chronological rollout buffers.
        """
        if term_obs:
            if steps != 1:
                raise NotImplementedError("term_obs sampling requires steps=1.")
            if "steps_to_go" not in self._td.keys(True, True):
                raise RuntimeError(
                    "term_obs requires compute_return() first (missing 'steps_to_go')."
                )

        if len(self) == 0 or self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")
        
        batch_size = int(batch_size)
        idx_flat = torch.randint(
            0, self.num_samples, ((batch_size),), device=self._td.device
        )

        t, e = torch.unravel_index(idx_flat, (len(self), self._td.shape[1]))
        
        if self.fake_bootstrap and next_obs:
            # repeat the last observation for terminated/truncated states
            ts = (t.unsqueeze(0) + torch.arange(steps + 1, device=self._td.device).unsqueeze(1)) % len(self)
            samples = self._td[ts, e]
            observations = samples.select(*self.observation_keys)
            observations_t = observations[:steps]
            observations_t1 = observations[1:]
            # tensordict supports `torch.where`
            tensordict_next = torch.where(
                samples[:steps]["next", "done"].squeeze(-1), # [steps, batch_size, 1] -> [steps, batch_size]
                observations_t, # [steps, batch_size]
                observations_t1, # [steps, batch_size]
            )
            samples = samples[:steps]
            samples["next"].update(tensordict_next)
        else:
            if steps == 1:
                samples = self._td[t, e]#.rename("env")
            else:
                ts = (t.unsqueeze(0) + torch.arange(steps, device=self._td.device).unsqueeze(1)) % len(self)
                samples = self._td[ts, e]#.rename("time", "env")
                assert samples.shape[:2] == (steps, batch_size)

        if term_obs:
            # Terminal transition is ``steps_to_go - 1`` steps ahead (inclusive horizon).
            stg = samples["steps_to_go"].squeeze(-1).long().clamp_min(1)
            term_t = (t + stg - 1).clamp_max(len(self) - 1)
            samples.set(
                "term_obs",
                self._td.select(*self.observation_keys)[term_t, e],
            )

        return samples

    def sample_trajectory(
        self,
        batch_size: int,
        steps: int,
        next_obs: bool = False,
    ) -> TensorDict:
        """Sample chronologically contiguous segments without crossing the write seam.

        Starts are drawn in logical (oldest-to-newest) ring order and then mapped
        to physical storage indices.  Unlike :meth:`sample`, a segment can never
        wrap from the newest stored transition back to the oldest one.
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}.")
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        # fake_bootstrap reconstructs s_{t+1} from the following stored row, so
        # it needs one additional chronological row beyond the returned segment.
        rows_needed = steps + int(self.fake_bootstrap and next_obs)
        if len(self) < rows_needed:
            raise RuntimeError(
                f"Need at least {rows_needed} stored rows for a length-{steps} trajectory, "
                f"but the buffer contains {len(self)}."
            )

        batch_size = int(batch_size)
        num_starts = len(self) - rows_needed + 1
        logical_start = torch.randint(
            0,
            num_starts,
            (batch_size,),
            device=self._td.device,
        )
        env_idx = torch.randint(
            0,
            self.num_envs,
            (batch_size,),
            device=self._td.device,
        )

        oldest = self._ptr if len(self) == self.max_size else 0
        physical_start = (oldest + logical_start) % self.max_size
        offsets = torch.arange(rows_needed, device=self._td.device).unsqueeze(1)
        physical_rows = (physical_start.unsqueeze(0) + offsets) % self.max_size
        samples = self._td[physical_rows, env_idx]

        if self.fake_bootstrap and next_obs:
            observations = samples.select(*self.observation_keys)
            observations_t = observations[:steps]
            observations_t1 = observations[1:]
            tensordict_next = torch.where(
                samples[:steps]["next", "done"].squeeze(-1),
                observations_t,
                observations_t1,
            )
            samples = samples[:steps]
            samples["next"].update(tensordict_next)

        assert samples.shape[:2] == (steps, batch_size)
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
    
    def compute_return(self, gamma: float, reward_collate_fn: Callable[[TensorDict | torch.Tensor], torch.Tensor]) -> None:
        """Compute return-to-go and steps-to-go for each filled step.

        Only valid for chronologically stored rollouts (e.g. :meth:`from_rollout`):
        episode ends must be visible in the buffer. Writes:

        * ``G`` — discounted Monte Carlo return-to-go
        * ``steps_to_go`` — steps until episode end (inclusive)

        If ``("next", "discount")`` is present with shape ``[T, N, 1]``, it multiplies
        ``gamma`` on each step: ``G_t = r_t + γ · discount_t · (1 - done_t) · G_{t+1}``.
        """
        if "G" in self._td.keys(True, True):
            raise ValueError("Return key 'G' already exists in buffer.")
        if "steps_to_go" in self._td.keys(True, True):
            raise ValueError("Key 'steps_to_go' already exists in buffer.")

        T = len(self)
        if T == 0:
            raise RuntimeError("Cannot compute returns on an empty ReplayBuffer.")
        # from_rollout stores oldest→newest in ``[:T]``; a wrapped online ring does not.
        if T == self.max_size and self._ptr != 0:
            raise RuntimeError(
                "compute_return requires chronological storage (e.g. from_rollout); "
                "wrapped ring buffers are not supported."
            )

        N = self.num_envs
        rew_raw = self._td[("next", "reward")] # must exist
        rew_raw = rew_raw[:T]
        rew = reward_collate_fn(rew_raw)

        done = self._td[("next", "done")] # must exist
        done = done[:T].float()
        if done.shape != (T, N, 1):
            raise ValueError(f"Expected done shape {(T, N, 1)}, got {tuple(done.shape)}.")

        discount = self._td.get(("next", "discount"), default=None)
        if discount is None:
            discount = torch.ones(T, N, 1, device=rew.device, dtype=rew.dtype)
        else:
            discount = discount[:T].float()
            if discount.shape != (T, N, 1):
                raise ValueError(
                    f"Expected discount shape {(T, N, 1)} or absent, got {tuple(discount.shape)}."
                )

        is_init = self._td.get("is_init", default=None)
        if is_init is not None:
            is_init = is_init[:T].bool()
            if is_init.shape == (T, N, 1):
                is_init = is_init.squeeze(-1)
            if is_init.shape != (T, N):
                raise ValueError(
                    f"Expected is_init shape {(T, N)} or {(T, N, 1)}, got {tuple(is_init.shape)}."
                )

        G = torch.zeros_like(rew)
        steps_to_go = torch.zeros(T, N, 1, device=rew.device, dtype=torch.long)
        running_g = torch.zeros(N, 1, device=rew.device, dtype=rew.dtype)
        running_h = torch.zeros(N, 1, device=rew.device, dtype=torch.long)

        for t in reversed(range(T)):
            cont = 1.0 - done[t]
            running_g = rew[t] + float(gamma) * discount[t] * cont * running_g
            running_h = 1 + (running_h * cont.long())
            G[t] = running_g
            steps_to_go[t] = running_h
            if is_init is not None:
                keep = (~is_init[t]).unsqueeze(-1)
                running_g = running_g * keep.float()
                running_h = running_h * keep.long()

        out_g = torch.zeros(
            self.max_size, N, 1, device=rew.device, dtype=rew.dtype
        )
        out_h = torch.zeros(
            self.max_size, N, 1, device=rew.device, dtype=torch.long
        )
        out_g[:T] = G
        out_h[:T] = steps_to_go
        self._td.set("G", out_g)
        self._td.set("steps_to_go", out_h)

    def __len__(self):
        return self._current_size


def format_nbytes(nbytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(nbytes)
    unit = units[0]
    for next_unit in units[1:]:
        if size < 1024.0:
            break
        size /= 1024.0
        unit = next_unit
    return f"{size:.2f}{unit}"
