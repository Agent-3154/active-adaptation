# Guideline (for humans) and Instruction (for agents) for Implementing a New Algorithm

This is a research-oriented codebase. We prefer single-file implementations that are conceptually clean and resemble the pseudocode as much as possible.

**Templates by training loop:**

| Loop | Script | Reference implementation |
|------|--------|--------------------------|
| On-policy | `scripts/train_ppo.py` | `learning/ppo/ppo.py`, `learning/ppo/ppo_symaug.py` |
| Off-policy | `scripts/train_offpolicy.py` | `learning/offpolicy/sac.py` (scalar), `sac_dist.py` (C51) |

This doc is WIP. Refine it as we add algorithms.

---

## General style

- Prefer a single algorithm file; extract helpers only for low-level math primitives that are reused widely.
- Do not use inheritance for algorithm classes (e.g. do not subclass `PPOBase` / `PPOPolicy` for a new PPO variant).
- Use type hints and short docstrings on non-obvious methods.
- Add informative diagnostics (e.g. explained variance, KL, Q stats) and brief comments where behavior is easy to misread.
- Start from the original paper/codebase hyperparameters and architecture when reproducing; layer in codebase conventions (Mish, ortho init, Muon) incrementally.
- For agents: leave comments when uncertain about an implementation choice.

---

## Shared conventions

### Hydra config

- Keep a dedicated algorithm file and register a dataclass config via `ConfigStore` in the same file (`algo=<name>`).
- Avoid `Literal[...]` on Hydra dataclass fields; use `str` / primitives in the schema and validate allowed values at runtime.
- Set `_target_` to the policy class (e.g. `active_adaptation.learning.offpolicy.sac.SAC`).

### TorchRL / TensorDict

- `TensorDictModule` accepts not only `nn.Module`, but also callables.
- Use TorchRL's `interaction_type` (`InteractionType.MODE` vs stochastic) inside rollout policies to switch eval vs exploration without duplicating policy code. `train_offpolicy.py` wraps rollout in `ExplorationType.RANDOM`; eval uses `get_rollout_policy("eval")` with `ExplorationType.MODE`.
- If rollout needs persistent state (RNN, AR(1) noise, etc.), implement `make_tensordict_primer()` so `make_env_policy` can attach a `TensorDictPrimer` **before** the env is used. The primer spec must match what the rollout policy reads/writes (see `sac.py`: `prev_noise`, `rho`).

### Diagnostics and stabilization

- Emit diagnostics tied to new behaviors so training is visible in logs.
- Expose stabilization knobs in config with safe defaults (advantage clamp, log-ratio clamp, etc.); keep them easy to disable.
- If symmetry augmentation is used, gate it with a config flag and ensure metrics still work when it is off.

---

## On-policy algorithms (`train_ppo.py`)

### Interface

The training script uses a **stacking collector**: it gathers `train_every` steps per env, then calls `policy.train_op(data)`.

| Method | Role |
|--------|------|
| `from_env(cls, cfg, env, device)` | Construct policy from env specs |
| `on_stage_start(stage, env)` | Stage setup (buffers, schedules, etc.) |
| `get_rollout_policy(mode)` | `TensorDictModuleBase` used inside the collector |
| `train_op(tensordict)` | One update on a stacked batch `[num_envs, train_every, …]` |
| `state_dict()` / `load_state_dict()` | Checkpoints |

Reuse the existing pipeline shape first (`train_op`, `_update`, `compute_advantage`, `get_rollout_policy`), then add algorithm-specific terms in `_update`.

### PPO-style extras

- For multiple trust-region variants, prefer one config switch (e.g. `trust_region_mode in {ppo, spo, aspo}`) over scattered objective branches.
- Sync running normalizers across ranks inside `train_op` when distributed (`VecNorm.synchronize`).

---

## Off-policy algorithms (`train_offpolicy.py`)

### Interface (`sac.py` / `sac_dist.py`)

The training script does **not** stack transitions. Each env step: rollout → `env.step_and_maybe_reset` → `policy.step(td)`.

| Method | Role |
|--------|------|
| `from_env(cls, cfg, env, device)` | Construct policy; optional symmetry transforms for `sym_aug` |
| `make_tensordict_primer()` | Optional; register rollout state keys on the env |
| `on_stage_start(stage, env)` | Build `ReplayBuffer` from `env.fake_tensordict()`; optional prior buffer (RLPD) |
| `get_rollout_policy(mode, critic=False)` | Per-step rollout policy (`SACRolloutPolicy` in `sac.py`) |
| `step(tensordict)` | Push transition to replay buffer; update running reward stats; call `train_op()` on schedule |
| `train_op()` | Sample buffer, run UTD critic steps + actor steps; return metrics dict |
| `state_dict()` / `load_state_dict()` | Checkpoints; use `unwrap_ddp` on wrapped modules |

`train_offpolicy.py` strips `next` observations and private `_` keys before `policy.step`. Do not rely on `next` obs being present in the buffer push.

Scalar vs distributional: `algo=sac` uses twin scalar critics; `algo=sac_dist` uses twin C51 critics. Both share `NormalActor`, `CriticTrunk`, `SimpleDoubleCritic`, and `SACRolloutPolicy` from `sac.py`. Leave experimental dual-stream SAC in `sac2.py`.

### Replay and batching

- Declare `train_keys` on the policy class; `select(*train_keys)` in `train_critic` / `train_actor`.
- Create the buffer in `on_stage_start`, not in `__init__`, so eval-only runs do not allocate replay memory.
- Use `ReplayBuffer.sample(..., steps=n_steps, next_obs=True)` for multi-step TD; keep `MultiStepReturn` in `objectives.py` for `n_steps > 1`.
- Warm-up: only push to the buffer until `global_step > warm_up_steps`; gate `train_op` on `global_step % train_every == 0`.

### Training loop shape (`sac.py`)

- **Critic:** `train_every * utd_ratio` updates per `train_op`; diagnostics on the last UTD step only.
- **Actor:** `train_every` updates per `train_op`; diagnostics on the last actor step only.
- Decorate `train_op` with `@VecNorm.freeze()` so observation stats are not updated during replay sampling.
- Before sampling, sync per-rank running stats: `vecnorm_obs.synchronize(mode="broadcast")` and `reward_normalizer.synchronize(mode="broadcast")` when enabled.

### Rollout policy patterns

- Preprocess with `self.preproc` (concat CMD+OBS, `VecNorm`) before the actor.
- Store `loc` in the tensordict when useful for diagnostics or RLPD comparisons.
- Correlated exploration: AR(1) pre-tanh noise via `prev_noise` / `rho` (primer + rollout policy); target policy uses uncorrelated samples in `_compute_target`.
- Optional `reward_normalizer`: update stats in `step` on raw rewards; normalize in `train_critic`; denormalize Q for logging.

### Distributed training

- Wrap **only** online `actor` and `Q` in DDP, **after** `deepcopy` of targets and **after** optimizers are built (DDP shares parameter tensors with the optimizer).
- Target nets stay plain modules; load targets with `unwrap_ddp(self.Q).state_dict()`.
- Broadcast all parameters/buffers (including targets, `vecnorm_obs`, `log_alpha`) from rank 0 at startup.
- `log_alpha` is not inside DDP; all-reduce its gradient manually after `alpha_loss.backward()`.
- Checkpoints: save/load via `unwrap_ddp` so single-GPU and DDP checkpoints are interchangeable.

### Performance notes

- `torch.compile` the target computation (`_compute_target`) only; compiling the full critic loss was numerically inconsistent / slower as of torch 2.11.
- With AMP: `grad_scaler.unscale_` before grad clip; keep `log_alpha` / alpha update in fp32.

### Optional features (see `sac.py` / `sac_dist.py`)

- **RLPD:** `prior_data` + `prior_data_ratio`; concat prior batch in `train_critic` / `train_actor`; log `critic/prior_q_*`, `actor/online_advantage`.
- **Symmetry aug:** duplicate `(obs, act, targets)` with `obs_transform` / `act_transform` in the learner.
- **Distributional critic:** `algo=sac_dist` / `C51Critic` via `distributional.py`; tune `v_min` / `v_max` with `normalize_reward` if used.
- **Reward normalization:** `RewardNormalizer` in `reward_normalization.py` (FlashSAC-style); buffer stores raw rewards.

---

## Case study: FPO++ (`learning/ppo/fpo.py`)

Notes from adding a flow-matching PPO variant:

- Rollout must emit all ratio bookkeeping tensors explicitly (`cfm_loss`, `cfm_loss_eps`, `cfm_loss_t`, etc.) so `train_op` stays on replayed tensors only.
- CFM ratio: `log_ratio = old_cfm_loss - new_cfm_loss` using the **same** `(eps, t)` stored at rollout — not Gaussian log-prob surrogates.
- Flow actor: integrate with consistent time convention; store CFM samples at rollout for the trust-region update.
- Time conditioning: prefer modern timestep embedding + FiLM (`ConditionalBlock`) over naive concat.
- Trust region: reuse `fpo_surrogate_loss` with `trust_region_mode in {ppo, spo, aspo}`.
- Diagnostics: `actor/approx_kl`, clamp fraction, `cfm_loss` old vs new, `critic/explained_var` as applicable.
- DDP: wrap the module actually used in training forwards (`flow_actor_impl` / rollout module), not a stale `self.actor` reference.
