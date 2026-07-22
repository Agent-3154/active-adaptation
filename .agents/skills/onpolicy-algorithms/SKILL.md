---
name: onpolicy-algorithms
description: Implement, extend, and run on-policy RL algorithms in active-adaptation (PPO, symmetry augmentation, SPO, Muon). Use when adding or modifying algo files under learning/ppo, wiring Hydra algo configs, debugging train_ppo.py runs, GAE/advantage computation, or trust-region policy updates.
---

# On-Policy Algorithms (active-adaptation)

Research-oriented codebase: **one algorithm file per variant**, pseudocode-shaped, **no inheritance between algo classes** (do not subclass `PPOBase` for new variants — copy `ppo_symaug.py` and edit).

**Primary references**
- Training loop: `scripts/train_ppo.py`
- Canonical PPO + sym aug: `learning/ppo/ppo_symaug.py`
- Simpler baseline: `learning/ppo/ppo.py` (uses `PPOBase`, `symnet`/`symaug` flags)
- Shared losses/keys: `learning/ppo/common.py`
- Human/agent guide: `learning/TEACHME.md` (on-policy section)

Read `reference.md` in this skill for file map and loop diagram.

**Related skills:** `offpolicy-algorithms` (replay-based SAC/TD3), `environment-mdp` (tasks / MDP terms).

---

## When to use this skill

- Adding a new on-policy algorithm (PPO variant, FPO, trust-region tweak)
- Porting symmetry augmentation or Muon optimizer patterns
- Debugging GAE, clipped surrogate, KL/LR adaptation, DDP grad sync
- Choosing `algo=ppo` vs `algo=ppo_symaug` for recipes

---

## Quick start (running)

```bash
# from active-adaptation/
python scripts/train_ppo.py task=<Task> algo=ppo_symaug
python scripts/train_ppo.py task=A2/A2LocoManip algo=ppo_symaug algo.muon=true
```

PPO is often stage 1 in RLPD recipes (`cfg/recipe/*_rlpd.yaml`) before `rollout.py` → `train_offpolicy.py`.

---

## Hydra config pattern

`_target_` is the **config dataclass**; `get_class()` returns the policy. `helpers.make_env_policy` instantiates the config (runs `__post_init__`) then calls `from_env`. See `TEACHME.md` (Hydra config). Example: `ppo_symaug.PPOConfig`, `ppo_teacher_student.PPOTSCfg`.

---

## `ppo.py` vs `ppo_symaug.py`

| | `ppo.py` | `ppo_symaug.py` (preferred template) |
|---|----------|--------------------------------------|
| Base class | `PPOBase` | `TensorDictModuleBase` (standalone) |
| Symmetry | `symnet` (wrapper) or `symaug` flag | Always augments in `_augment_symmetry` during update |
| Trust region | PPO clip only | `spo` flag → SPO loss; else `ppo_clipped_loss` |
| Optimizer | AdamW / optional Muon | AdamW / `MuonAdamWWrapper` |
| DDP | via base | `on_stage_start`: wrap actor+critic in DDP |
| Extras | simpler | aux head, effective rank, dormancy debug |

**Default for new work:** start from `ppo_symaug.py`.

---

## Required policy interface (`train_ppo.py`)

Uses a **stacking collector** (`BufferCollector`): gathers `train_every` steps per env, then `policy.train_op(data)` on shape `[num_envs, train_every, …]`.

| Method | Role |
|--------|------|
| `from_env(cls, cfg, env, device)` | Build policy; wire symmetry transforms from env |
| `on_stage_start(stage, env)` | DDP wrap, optimizer, optional `torch.compile` on `_update` |
| `get_rollout_policy(mode, critic=False)` | `Seq(vecnorm, actor)` or `+ critic`; rollout stores `action_log_prob` |
| `train_op(tensordict)` | GAE → normalize adv → epochs × minibatches → `_update` |
| `state_dict()` / `load_state_dict()` | Checkpoints; unwrap DDP modules |

No replay buffer. All training tensors come from the collected rollout batch.

---

## Implementing a new algorithm (checklist)

### 1. File + Hydra

- Add `learning/ppo/<name>.py`
- Dataclass: `_target_` = **config class path**, `get_class()` → policy class, `name`, `cs.store(..., group="algo")`
- No `Literal[...]` in Hydra fields — validate at runtime
- Derived fields (e.g. union `in_keys`) in `__post_init__` as **tuple/list**, never `set`
- `ppo/__init__.py` auto-imports `*.py`

### 2. Network stack

```python
# Typical layout (ppo_symaug)
vecnorm: CatTensors(CMD+OBS) → VecNorm → "_obs_normed"
actor:   MLP → Actor → ProbabilisticActor(IndependentNormal, return_log_prob=True)
critic:  MLP → Critic(1) → "state_value"
```

- `training_keys`: tensors needed in `_update` (`action_log_prob`, `adv`, `ret`, `is_init`, obs/cmd/action)
- Orthogonal init on linear layers; small init on actor mean (see `ppo_symaug` `init_`)

### 3. `train_op` pipeline

1. `@VecNorm.freeze()` on `train_op` (obs stats fixed during update)
2. `compute_advantage(tensordict, critic, "adv", "ret", clamp_reward)`
3. Normalize advantages: `(adv - mean) / std.clamp_min(1e-7)`
4. `td = tensordict.select(*training_keys)`
5. For `ppo_epochs` × `num_minibatches`: optional sym aug → `_update(minibatch)`
6. Optional adaptive LR from `desired_kl` (see `ppo_symaug`)
7. Post-update diagnostics (policy gain, explained var, effective rank)
8. If distributed: `vecnorm.apply(vecnorm_sync_)` after update

### 4. `_update` (one minibatch)

Pattern in `ppo_symaug._update`:
- `vecnorm(minibatch)` → forward actor on stored actions → `log_ratio = log π_new - log π_old`
- `policy_loss = ppo_clipped_loss(ratio, adv, clip_param)` or `spo_loss` if `cfg.spo`
- `value_loss = MSE(ret, V(s))` masked by `~is_init`
- `loss = policy_loss - entropy_coef * entropy + value_loss` (+ optional aux)
- `backward` → manual grad all-reduce if `should_reduce_grads` else DDP
- `clip_grad_norm_` on actor and critic separately
- Return per-minibatch metrics dict (aggregated in `train_op`)

### 5. GAE (`compute_advantage`)

- Aggregate reward dict → scalar `reward_aggregated`
- Optional `clamp_reward` (min 0) before GAE
- Scale rewards by `(1 - gamma)` for effective-horizon normalization
- Use `GAE(0.99, 0.95)` from `common.py` with `values`, `next_values`, `discount`, `terminated`, `done`
- Write `adv` and `ret` into tensordict

### 6. Rollout policy

- `ProbabilisticActor` must set `return_log_prob=True` so collector stores `action_log_prob`
- `get_rollout_policy`: `Seq(self.vecnorm, self.actor)` — no sym aug at rollout (aug only in learner)
- Eval: `train_ppo.py` uses `ExplorationType.MODE` for eval rollouts

### 7. Symmetry augmentation (`ppo_symaug` style)

- `from_env`: `cmd_transform`, `obs_transform`, `act_transform` from env observation groups / action manager
- `_augment_symmetry`: mirror batch, concat `[original, mirrored]`; copy `adv`, `ret`, `log_prob`, `is_init` unchanged
- `_update` assumes doubled batch (`bsize = shape[0] // 2`); log `actor/symmetry_loss`

Alternative in `ppo.py`: `SymmetryWrapper` around actor/critic when `symnet=True`.

### 8. Distributed training

**Default: use DDP** (`use_ddp=True` in `ppo_symaug`). Manual gradient `all_reduce` in `_update` is only when `use_ddp=False` (`should_reduce_grads`).

#### What to wrap

| Wrap in DDP (`on_stage_start`) | Do **not** wrap |
|--------------------------------|-----------------|
| `actor`, `critic` | `vecnorm` (sync running stats separately) |
| | Symmetry transforms (stateless / separate objects) |

Wrap in `on_stage_start`, then build the optimizer — DDP shares underlying parameter tensors, so `AdamW(self.actor.parameters(), …)` remains valid.

```python
if self.cfg.use_ddp:
    self.actor = DDP(self.actor, device_ids=[aa.get_local_rank()])
    self.critic = DDP(self.critic, device_ids=[aa.get_local_rank()])
self.should_reduce_grads = aa.is_distributed() and not self.cfg.use_ddp
```

Prefer `wrap_ddp` from `learning/utils/distributed.py` in new code (attribute forwarding + typing); `ppo_symaug` uses raw `DDP` today.

#### Manual grad sync (fallback only)

When `should_reduce_grads`:

```python
for param in self.actor.parameters():
    distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
    param.grad /= self.world_size
# same for critic
```

Do **not** combine this with DDP on the same module (double reduction → wrong effective LR).

#### Running statistics

After `train_op`, sync observation normalization across ranks:

```python
self.vecnorm.apply(vecnorm_sync_)  # VecNorm.synchronize(mode="broadcast")
```

Each rank collects its own rollout shard in `BufferCollector`; advantages are computed per-rank on that shard — no replay sync needed.

#### Checkpoints

`state_dict()`: unwrap DDP before saving (`module.module` if `isinstance(module, DDP)`). Checkpoints must load on single-GPU and multi-GPU.

#### Caveats / common bugs

- **`use_ddp=True` + manual `all_reduce`** — remove one; DDP already averages grads in backward.
- **Wrapping `vecnorm`** — running mean/var are per-rank unless you `synchronize`; we sync explicitly instead of DDP-wrapping.
- **`torch.compile`** — disabled when `aa.is_distributed()` in `ppo_symaug`; compiling rollout/update under DDP is fragile.
- **Optimizer param groups** — actor and critic are separate groups; adaptive LR in `train_op` only touches `param_groups[0]` (actor).
- **Rollout policy** — `get_rollout_policy` uses `self.actor` (DDP-wrapped); forward works; for eval-only exports use `unwrap_ddp(self.actor)`.
- **Debug** — `check_parameters(self.actor)` in `debug` mode verifies cross-rank param agreement (`learning/utils/distributed.py`).

Launch: `scripts/launch_ddp.py 0,1 scripts/train_ppo.py …` (see `AGENTS.md`).

### 9. Trust-region variants

Prefer one config switch over scattered branches:
- `spo: bool` → `spo_loss` vs `ppo_clipped_loss` (`common.py`)
- `clip_param`: scalar ε or `(eps_neg, eps_pos)` via `resolve_clip_param`
- For FPO++ see `learning/ppo/fpo.py` — store CFM bookkeeping tensors at rollout

---

## Config knobs (common overrides)

```yaml
algo=ppo_symaug
algo.train_every=32          # rollout steps per update (per env)
algo.ppo_epochs=4
algo.num_minibatches=4
algo.lr=5e-4
algo.muon=true
algo.spo=false
algo.clip_param=[0.2, 0.2]   # asymmetric clip OK
algo.desired_kl=0.01         # adaptive LR; null to disable
algo.entropy_coef=0.002
algo.clamp_reward=false
algo.compile=false           # avoid with debug or DDP
```

---

## Diagnostics to emit

- `actor/policy_loss`, `actor/entropy`, `actor/grad_norm`, `actor/approx_kl`
- `actor/clamp_pos`, `actor/clamp_neg` (clip fraction)
- `critic/value_loss`, `critic/explained_var`, `critic/grad_norm`
- `critic/adv_mean`, `critic/adv_std`, `critic/value_mean`
- Symmetry: `actor/symmetry_loss`
- Post-update: `actor/policy_gain`, `actor/weighted_ratio`, `actor/effective_rank`

---

## Anti-patterns

- Subclassing `PPOPolicy` / `PPOBase` for a new algorithm file
- Updating `VecNorm` during `_update` without `@VecNorm.freeze()` on `train_op`
- Symmetry-aug at rollout instead of in the learner (doubles env cost; not our pattern)
- `torch.compile` with `debug=True` or distributed (guarded in `ppo_symaug`)
- Forgetting `action_log_prob` in rollout (breaks importance ratio)
- Using off-policy `step()` / replay patterns in on-policy code
- Building twin actors with `copy.deepcopy` when the graph still has `Lazy*` params (e.g. `Actor`) — use a `make_actor()` factory twice; see TEACHME “Cloning modules”
- Freezing teacher modules with `requires_grad_(False)` during distill and forgetting to re-enable them before the next PPO update
- Letting `VecNorm` keep updating in a student/adapt stage while the actor is frozen (freeze norms after the teacher checkpoint)

---

## Module clones and lazy init

- Weight sync: `hard_copy_` / `soft_copy_` in `learning/ppo/common.py` (strict: names, shapes, no shared storage, no `Uninitialized*`).
- Teacher–student (`ppo_teacher_student.py`): `make_actor()` twice for `actor_teacher` / `actor_student`; `hard_copy_` on student stage start — not `deepcopy` of the teacher.
- **Deprecate lazy init going forward** — prefer explicit dims from specs; do not add new `LazyLinear` / lazy MLP entry points.

---

## Verification

1. Import + Hydra `algo=<name>` resolves
2. Short `train_ppo.py` run: metrics move, no NaN KL/grad norms
3. Checkpoint round-trip
4. If `sym_aug`: `actor/symmetry_loss` logged, batch doubling correct
5. DDP: `vecnorm` synced, actor/critic params match rank 0 after broadcast
