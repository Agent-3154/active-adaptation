---
name: offpolicy-algorithms
description: Implement, extend, and run off-policy RL algorithms in active-adaptation (SAC, distributional SAC, SimbaV2, RLPD, BAC/BEE). Use when adding or modifying algo files under learning/offpolicy, wiring Hydra algo configs, debugging train_offpolicy.py runs, replay buffers, or critic/actor training loops.
---

# Off-Policy Algorithms (active-adaptation)

Research-oriented codebase: **one algorithm file per variant**, pseudocode-shaped, minimal inheritance.

**Primary references**
- Training loop: `scripts/train_offpolicy.py`
- Canonical SAC: `active_adaptation/learning/offpolicy/sac.py`
- Human/agent guide: `active_adaptation/learning/TEACHME.md` (off-policy section)

Read `reference.md` in this skill for the algorithm catalog and shared-module map.

**Related skills:** `onpolicy-algorithms` (PPO / `train_ppo.py`), `environment-mdp` (tasks / MDP terms).

---

## When to use this skill

- Adding a new off-policy algorithm or SAC variant
- Porting a paper (SAC, C51, BAC/BEE, SimbaV2, RLPD, TD3, …)
- Debugging replay sampling, targets, AMP/NaN grads, DDP checkpoints
- Choosing `algo=…` overrides for recipes / Hydra runs

---

## Quick start (running)

```bash
# from active-adaptation/
python scripts/train_offpolicy.py task=<Task> algo=sac
python scripts/train_offpolicy.py task=A2/A2LocoManip algo=sac_dist algo.prior_data=/path/to/rollout.pt
```

Multi-stage pipelines: see `cfg/recipe/*_rlpd.yaml` (PPO → rollout → `train_offpolicy.py` with `algo.prior_data`).

Hydra configs live under `cfg/`; algorithms register via `ConfigStore` in their own file (`algo=<name>`).

### Hydra config pattern

`_target_` → **config dataclass** (not the policy). Add `get_class(self)` returning the policy class so `helpers.make_env_policy` can `instantiate` the config (runs `__post_init__`) then `from_env`. See `TEACHME.md`. Canonical: `sac.SACConfig.get_class() → SAC`.

---

## Algorithm variants (pick one, don't fork blindly)

| `algo` | Critic | Actor / trunk | Optimizer notes |
|--------|--------|---------------|-----------------|
| `sac` | twin scalar | `NormalActor`, `CriticTrunk` | Muon+AdamW default; BAC/BEE optional |
| `sac_dist` | twin C51 | shared from `sac.py` | distributional targets in `distributional.py` |
| `sac_simba` | twin scalar | `SimbaV2Actor`, `SimbaV2CriticTrunk` | Adam, no WD/Muon; `normalize_hyper_dense_` after steps |
| `sac_dist_simba` | twin C51 | SimbaV2 nets | same as `sac_simba` |
| `sac2` | experimental dual-stream | — | leave alone unless explicitly asked |
| `lql_sac`, `td3`, `td31` | see file | — | specialized; read file before changing |

**Shared exports from `sac.py`:** `NormalActor`, `CriticTrunk`, `SimpleDoubleCritic`, `TwinScalarCritic`, `SACRolloutPolicy`, `AlphaModule`, `quantile_mse_loss`.

---

## Required policy interface (`train_offpolicy.py`)

The script does **not** stack transitions. Each env step: rollout → `env.step_and_maybe_reset` → `policy.step(td)`.

| Method | Role |
|--------|------|
| `from_env(cls, cfg, env, device)` | Build policy; wire `sym_aug` transforms if needed |
| `make_tensordict_primer()` | Optional rollout state (`prev_noise`, `rho` for AR(1) SAC noise) |
| `on_stage_start(stage, env)` | Create `ReplayBuffer` from `env.fake_tensordict()`; optional prior buffer (RLPD) |
| `get_rollout_policy(mode, critic=False)` | Per-step rollout (`SACRolloutPolicy` pattern) |
| `step(tensordict)` | Push transition; update reward stats; call `train_op()` on schedule |
| `train_op()` | Sample replay; UTD critic updates + actor updates; return metrics dict |
| `state_dict()` / `load_state_dict()` | Checkpoints via `unwrap_ddp` |

**Critical:** `train_offpolicy.py` strips `next` observations and `_` private keys before `policy.step`. Do not rely on `next` obs in the buffer push path.

---

## Implementing a new algorithm (checklist)

### 1. File + Hydra registration

- Add `active_adaptation/learning/offpolicy/<name>.py`
- Dataclass config with `_target_`, `name`, `cs.store(name="<name>", node=..., group="algo")`
- No `Literal[...]` in Hydra fields — validate strings at runtime
- `offpolicy/__init__.py` auto-imports all `*.py` files → config registers on import

### 2. Class shape

```python
class MyAlgo(TensorDictModuleBase):
    train_keys = (CMD_KEY, OBS_KEY, ("next", OBS_KEY), ...)

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, ...):
        # build nets, targets (deepcopy BEFORE DDP), optimizers, reward_normalizer
        # if distributed: wrap actor/Q after optimizers; broadcast params

    def train_critic(self, batch, batch_prior=None, diagnostics=False): ...
    def train_actor(self, diagnostics=False): ...
    def train_op(self): ...  # UTD loop + actor loop; @VecNorm.freeze()
```

Prefer copying `sac.py` structure and deleting/changing only what the paper requires.

### 3. Replay + batching

- Declare `train_keys`; `batch.select(*train_keys)` in critic/actor
- Buffer in `on_stage_start`, not `__init__` (eval-only runs must not allocate)
- `ReplayBuffer.sample(..., steps=n_steps, next_obs=True)` for multi-step TD; use `MultiStepReturn` when `n_steps > 1`
- Warm-up: push only until `global_step > warm_up_steps`; train when `global_step % train_every == 0`
- Gate diagnostics: last UTD critic step and last actor step only (see `sac.py`)

#### `buffer_size` meaning (easy to get wrong)

`cfg.buffer_size` is **not** a flat transition count. `ReplayBuffer.from_fake(max_size, fake_tensordict)` allocates a ring with batch shape **`[buffer_size, num_envs]`**:

| Quantity | Value |
|----------|--------|
| Ring length (time / steps) | `buffer_size` (e.g. SAC default **2000**) |
| Parallel slots | `num_envs` (often thousands) |
| Max stored transitions | `buffer_size * num_envs` |
| One `push(td)` | writes **one step** across all envs at `write_ptr` |

So `buffer_size=2000` with 4096 envs ≈ 8M transitions. Values like `1e5`–`1e6` (common in single-env / flat-buffer papers) are **far too large** here and will OOM. Prefer O(10³) ring lengths aligned with SAC (`2000`) / TD3 (`1500`), not paper “1e6 replay size” numbers.

`len(rb)` / `rb.max_size` report the filled/capacity **ring length** (first dim), not `× num_envs`. `rb.num_samples == len(rb) * num_envs` is the transition count. Sampling draws `(t, env)` pairs uniformly over stored rows.

See `learning/offpolicy/buffer.py` (`from_fake`, `push`, `sample`) and `sac.py` `on_stage_start`.

### 4. Training loop counts (SAC convention)

- **Critic:** `train_every * utd_ratio` updates per `train_op`
- **Actor:** `train_every` updates per `train_op`
- Before sampling: `vecnorm_obs.synchronize(mode="broadcast")`, `reward_normalizer.synchronize(...)` if used

### 5. Rollout policy

- `self.preproc`: concat CMD+OBS → `VecNorm` → `_input_normed`
- Store `loc` in tensordict when useful for RLPD / diagnostics
- Correlated exploration: primer keys `prev_noise`, `rho`; target policy uses uncorrelated noise in `_compute_target`
- `reward_normalizer`: update on raw rewards in `step`; normalize in `train_critic`; denormalize Q for logging

### 6. Distributed training

**Default: use DDP** (`wrap_ddp` / `torch.nn.parallel.DistributedDataParallel`). Manual `dist.all_reduce` on gradients is a legacy fallback only when DDP is intentionally disabled.

#### What to wrap (online / trained modules only)

| Wrap in DDP | Do **not** wrap |
|-------------|-----------------|
| `actor`, `Q` (and `V`, `alpha` if present) | `Q_target`, `actor_target` (plain `deepcopy`, eval-only) |
| | `VecNorm`, `RewardNormalizer` (sync running stats separately) |

#### Target `deepcopy` and lazy init

- `deepcopy` targets **only after** a warm-up forward has materialized any `Lazy*` params (canonical SAC already does this). Copying still-lazy graphs is unsafe; `soft_copy_` / `hard_copy_` in `learning/ppo/common.py` reject `Uninitialized*` and shared storage.
- For **twin trainable** modules (rare off-policy; common in on-policy teacher–student), prefer constructing twice via a factory — see TEACHME “Cloning modules”.
- **Future:** deprecate lazy initialization codebase-wide; prefer explicit dims from specs in new nets.

#### Correct setup order (off-policy, `sac.py` pattern)

1. Build modules + `deepcopy` targets (targets stay **unwrapped**).
2. Build optimizers from `self.actor.parameters()`, `self.Q.parameters()`, etc.
3. **Then** `wrap_ddp` online modules (`actor`, `Q`, `alpha`, optional `V`).
4. `_broadcast_parameters()` from rank 0 — include targets, `vecnorm_obs`, and wrapped modules (targets are not DDP but must match across ranks after local `deepcopy`).

DDP reuses the same parameter tensors the optimizers already hold; wrapping after `optimizer` construction is intentional.

Use `learning/utils/distributed.py`:

```python
from active_adaptation.learning.utils.distributed import wrap_ddp, unwrap_ddp

self.actor = wrap_ddp(self.actor, broadcast_buffers=True, find_unused_parameters=False, ...)
```

`find_unused_parameters=False` unless you have genuinely unused forward paths (slower otherwise).

#### During training

- Before replay sampling each `train_op`: `vecnorm_obs.synchronize(mode="broadcast")`; same for `reward_normalizer` if enabled.
- `soft_copy_(self.Q, self.Q_target, …)` and target `load_state_dict` use **`unwrap_ddp(self.Q)`** — never load into a DDP wrapper by mistake.
- Simba: `normalize_hyper_dense_(unwrap_ddp(self.Q))` after optimizer steps.
- AMP: `grad_scaler.unscale_` **before** `clip_grad_norm_` (grad norm is meaningless on scaled grads).
- Do **not** add manual `all_reduce` on grads when modules are DDP-wrapped (double reduction).

#### Checkpoints

Save/load via `unwrap_ddp(module).state_dict()` so single-GPU and multi-GPU checkpoints are interchangeable.

#### Caveats / common bugs

- **Wrapping targets** — breaks `soft_copy_` / EMA semantics; targets must stay plain `nn.Module`.
- **Optimizers built on the wrong module** — if you reassign `self.Q = wrap_ddp(self.Q)` after `Adam(self.Q.parameters())`, you're fine (same tensors); if you rebuild the optimizer after wrap without reusing params, you get stale or empty optimizer state.
- **Accessing submodules** — use `unwrap_ddp` for `state_dict`, `normalize_hyper_dense_`, and target copies; `wrap_ddp` forwards attributes but `isinstance(m, DDP)` checks need the wrapper.
- **Running normers** — DDP does not sync `VecNorm` / `RewardNormalizer`; call `.synchronize(mode="broadcast")` explicitly.
- **`torch.compile`** — avoid compiling full critic loss; optional on `_compute_target` only; test carefully with DDP.
- **Per-rank replay** — each rank pushes its own env transitions; that is expected. Sync **statistics**, not replay contents.

Launch: `scripts/launch_ddp.py` / `launch_ddp.sh` → `torchrun` (see `AGENTS.md` pipelines).

### 7. Numerics

- Critic loss in **fp32** inside `ScalarCritic.compute_loss` (AMP-safe)
- Optional `critic_loss: mse | huber` + `huber_delta` on scalar SACs
- `grad_scaler.unscale_` before `clip_grad_norm_`
- Do **not** `torch.compile` full critic loss (unstable as of torch 2.11); `_compute_target` compile is optional
- Simba: call `normalize_hyper_dense_(unwrap_ddp(net))` after actor/Q (and V) optimizer steps

---

## Feature recipes (copy from `sac.py`)

### RLPD
`prior_data` + `prior_data_ratio` → second buffer; concat prior batch in `train_critic` / `train_actor`; log `critic/prior_q_*`, `actor/online_advantage`.

### Symmetry augmentation
`sym_aug=True` requires env symmetry transforms; duplicate `(obs, act, targets)` inside `train_critic`.

### Reward normalization
`RewardNormalizer` (`reward_normalization.py`): buffer stores **raw** rewards; scale at train time by running return variance (FlashSAC-style).

### BAC / BEE (scalar `sac`, `sac_simba`)
Reference: `lib/Seizing-Serendipity-.../model/algorithm.py`

- `learn_v=True`, `v_quantile` (τ>0.5 → optimistic V), `bee_lambda` (0 = plain SAC)
- **V loss:** quantile-regress `V(s) − α log π(a|s)` toward `min Q_target(s, a_replay)`
- **Q target:** `λ·(r + γ V(s')) + (1−λ)·soft-Q bootstrap`
- `bee_lambda > 0` requires `learn_v=True`
- If NaN critic grads under AMP+BEE: try `critic_loss=huber` or lower `bee_lambda`

### Distributional (C51)
- Use `C51Critic` + `TwinC51Critic` pattern in `sac_dist.py` / `sac_dist_simba.py`
- Tune `v_max`, `num_atoms` with `normalize_reward`
- Log `support_saturation_stats` when debugging support clipping

---

## Config knobs (common overrides)

```yaml
algo=sac
algo.train_every=4
algo.utd_ratio=4
algo.buffer_size=2000        # in episodes/transitions per env slot — read buffer.py
algo.warm_up_steps=200
algo.critic_batch_size=2048
algo.normalize_reward=true
algo.bee_lambda=0.5          # 0 disables BEE
algo.critic_loss=huber       # scalar SACs
algo.prior_data=/path/to.pt  # RLPD
algo.sym_aug=true
```

---

## Diagnostics to emit

Always log when adding behavior:
- `critic/q_loss`, `critic/grad_norm`, `critic/q_value`, `critic/q_max`
- `actor/loss`, `actor/alpha`, `actor/grad_norm`
- BEE: `critic/q_upper` (V mean)
- C51: support saturation / headroom stats
- RLPD: prior vs online Q gaps

---

## Anti-patterns

- Subclassing another algo class — copy and edit one file instead
- Creating replay buffer in `__init__`
- Relying on `next` obs inside `step()` after `train_offpolicy` strips them
- DDP-wrapping target networks
- Putting shared math in the algo file when it belongs in `distributional.py`, `objectives.py`, or `reward_normalization.py`
- Raising Simba LR / adding weight decay (paper uses ~1e-4 Adam, hyperspherical renorm)
- `deepcopy` of online nets **before** a warm-up forward when the graph still has `Lazy*` params — materialize first; `soft_copy_` rejects `Uninitialized*`
- Adding new lazy / `nn.LazyLinear` entry points — **lazy init is deprecated going forward**; prefer explicit dims from specs (see TEACHME)

---

## Verification before finishing

1. `python -c "from active_adaptation.learning.offpolicy import <module>; ..."` imports cleanly
2. Hydra: `algo=<name>` resolves without ConfigStore collision
3. Smoke: construct policy via `from_env` on a small task config
4. Checkpoint round-trip: `state_dict` → `load_state_dict`
5. If distributed: rank-0 broadcast + `unwrap_ddp` checkpoint portability
