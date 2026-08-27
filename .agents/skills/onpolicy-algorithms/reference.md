# On-policy reference

## Skill layout

Canonical source: `.agents/skills/` (committed; Cursor discovers this path).

```
.agents/skills/
├── onpolicy-algorithms/   # train_ppo.py, learning/ppo/
├── offpolicy-algorithms/  # train_offpolicy.py, learning/offpolicy/
└── environment-mdp/       # tasks, rewards, commands, obs, terms
```

Keep skills separate from each other and from `TEACHME.md` (shared style only).

## File map (`learning/ppo/`)

| File | Purpose |
|------|---------|
| `ppo_symaug.py` | **Primary template** — sym aug in learner, SPO/PPO, Muon, aux |
| `ppo.py` | Baseline PPO on `PPOBase`; `symnet` / `symaug` flags |
| `ppo_base.py` | Legacy shared base — do not subclass for new algos |
| `common.py` | Keys, GAE, PPO/SPO losses, `make_batch`, **`hard_copy_` / `soft_copy_`** (strict) |
| `ppo_teacher_student.py` | Two-stage TS; twin actors via `make_actor()` ×2 (not `deepcopy`); `PPOTSCfg` `_target_` + `get_class()` |
| `fpo.py` | Flow-matching PPO++ (CFM ratio tensors at rollout) |
| `ppo_rnn.py`, `ppo_him.py`, … | Specialized — read before editing |

## `train_ppo.py` loop

```
collector = BufferCollector(env, steps=algo.train_every)
for i in range(total_iters):
    data, carry = collector.collect(carry, rollout_policy)  # [N, train_every, ...]
    train_info = policy.train_op(data)
    vecnorm updated implicitly on rollout via collector's fresh obs
```

`frames_per_batch = num_envs * train_every`. Unlike off-policy, **no** per-step `policy.step`.

## `BufferCollector`

- Pre-allocates `[num_envs, train_every, …]` from `env.fake_tensordict()`
- Strips private `_` keys and `next` obs (same idea as off-policy script)
- Rollout under `ExplorationType.RANDOM`

## Advantage computation (GAE)

```python
rewards = sum reward dict components → [N, T, 1]
optional: rewards.clamp_min(0)
rewards *= (1 - gamma)
adv, ret = GAE(gamma, lam)(rewards, terminated, done, values, next_values, discount)
adv = (adv - adv.mean()) / adv.std().clamp_min(1e-7)  # in train_op
```

## Trust-region losses (`common.py`)

- **PPO:** `ppo_clipped_loss(ratio, adv, clip_param)` — minimize negative clipped surrogate
- **SPO:** `spo_loss(ratio, adv, clip_param)` — when `cfg.spo=True`
- **Clip param:** `resolve_clip_param` accepts scalar or `[eps_neg, eps_pos]`

## Muon optimizer

```python
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
self.opt = MuonAdamWWrapper([self.actor, self.critic], lr=lr, weight_decay=0.01)
```

MLP layers: `first_non_muon=True` on `MLP` marks input layers `_non_muon` for AdamW split.

## Symmetry transforms

From env at `from_env`:
```python
obs_transform = env.observation_groups[OBS_KEY].symmetry_transform()
act_transform = env.action_manager.symmetry_transform()
cmd_transform = env.observation_groups[CMD_KEY].symmetry_transform()  # if CMD present
```

Augmentation duplicates batch in `_augment_symmetry` before `_update`, not during env rollout.

## Case study pointers

- **FPO++** (`fpo.py`): rollout must emit `cfm_loss`, `cfm_loss_eps`, `cfm_loss_t`; ratio from stored CFM samples
- **RLPD pipeline**: PPO checkpoint → `rollout.py` → off-policy with `prior_data`

## Distributed training (summary)

| Topic | Guideline |
|-------|-----------|
| Default | **`use_ddp=True`**; manual `all_reduce` only if `use_ddp=False` |
| Wrap | `actor`, `critic` in `on_stage_start` |
| Never wrap | `vecnorm` (use `vecnorm_sync_` after `train_op`) |
| Fallback | `should_reduce_grads` → sum grads, divide by `world_size` |
| Checkpoints | Unwrap `DDP` in `state_dict()` |
| Compile | Off when distributed (`ppo_symaug`) |

See SKILL.md § “Distributed training” for caveats (double reduction, compile, param groups).

## Related docs

- `active_adaptation/learning/TEACHME.md` — general style + on/off-policy interfaces
- `.agents/skills/README.md` — how to use skills in Cursor vs other agents
- `.agents/skills/offpolicy-algorithms/` — SAC / replay counterpart
