# Off-policy reference

## File map (`learning/offpolicy/`)

| File | Purpose |
|------|---------|
| `sac.py` | Canonical scalar SAC + shared actor/critic/rollout pieces |
| `sac_dist.py` | C51 distributional SAC |
| `sac_simba.py` | Scalar SAC + SimbaV2 nets + BAC/BEE |
| `sac_dist_simba.py` | C51 + SimbaV2 |
| `sac2.py` | Experimental dual-stream SAC — do not refactor casually |
| `lql_sac.py` | Latent-Q SAC variant |
| `td3.py`, `td31.py` | TD3 variants |
| `buffer.py` | `ReplayBuffer` (ring buffer, multi-step sampling) |
| `distributional.py` | `ScalarCritic`, `C51Critic`, IQN/QR helpers, projection |
| `objectives.py` | `MultiStepReturn` |
| `reward_normalization.py` | `RewardNormalizer` |
| `noise.py` | Exploration noise helpers |

## `train_offpolicy.py` loop (simplified)

```
carry = env.reset()
for stage in stages:
    policy.on_stage_start(stage, env)
    rollout_policy = policy.get_rollout_policy("train")
    for i in range(total_iters):
        carry = rollout_policy(carry)          # ExplorationType.RANDOM
        td, carry = env.step_and_maybe_reset(carry)
        td = strip private keys and next obs
        train_info = policy.step(td)           # buffer push + maybe train_op
```

Inside `policy.step` (SAC pattern):
1. `rb.extend(td)` if past warm-up
2. `reward_normalizer.update(...)` on raw rewards
3. `global_step += 1`
4. if `global_step % train_every == 0`: `train_op()`

## `ReplayBuffer` essentials

- Built via `ReplayBuffer.from_fake(buffer_size, fake_td, ...)` in `on_stage_start`
- **Layout:** ring shape `[buffer_size, num_envs]` — `buffer_size` is steps along the ring, **not** total transitions (`capacity ≈ buffer_size * num_envs`). Defaults are O(10³) (e.g. SAC `2000`); do not copy flat-buffer paper sizes like `1e6`.
- `sample(batch_size, steps=n_steps, next_obs=True/False)`
- RLPD prior: `ReplayBuffer.from_rollout(path, ...)`
- Keys must match `train_keys` and env tensordict layout

## `ScalarCritic.compute_loss`

```python
pred = pred.float()
target = target.float().expand_as(pred)  # twin Q: pred [B,2], target [B,1]
# mse: (pred - target).square().sum(-1)
# huber: F.huber_loss(..., delta=huber_delta).sum(-1)
```

## BAC/BEE target (matches reference implementation)

```python
# expectile / soft-Q path
q_target_e = r + γ * (min Q_target(s', a') - α log π(a'|s'))

# direct V path
q_target_d = r + γ * V(s')

# mix
q_target = λ * q_target_d + (1 - λ) * q_target_e

# V regression (current state s, replay action a)
vf_pred = V(s) - α * log π(ã|s)   # ã ~ π, log π detached in our port
vf_target = min Q_target(s, a_replay)
vf_loss = quantile_mse_loss(vf_pred, vf_target, τ)
```

## SimbaV2 (`learning/modules/simba_v2.py`)

- `SimbaV2Actor`, `SimbaV2CriticTrunk` drop-in for `NormalActor` / `CriticTrunk`
- After each optimizer step on Simba modules: `normalize_hyper_dense_(unwrap_ddp(net))`
- Defaults: actor 256×1 block, critic 512×2 blocks, `lr=1e-4`, no Muon/WD

## Example recipe override chain

`cfg/recipe/a2_manip_rlpd.yaml`:
1. `train_ppo.py` — collect behavior prior
2. `rollout.py` — export transitions
3. `train_offpolicy.py` with `algo=sac_dist` and `algo.prior_data=${run_state.rollout_path}`

## Distributed training (summary)

| Topic | Guideline |
|-------|-----------|
| Default | **DDP** via `wrap_ddp`; no manual grad `all_reduce` on wrapped modules |
| Wrap | `actor`, `Q`, `alpha`, optional `V` |
| Never wrap | `Q_target`, `actor_target` |
| Order | `deepcopy` targets → build optimizers → `wrap_ddp` → `_broadcast_parameters` |
| Running stats | `vecnorm_obs.synchronize`, `reward_normalizer.synchronize` (not DDP) |
| Checkpoints | `unwrap_ddp(m).state_dict()` |
| Helpers | `learning/utils/distributed.py` |

Launch: `scripts/launch_ddp.py 0,1 scripts/train_offpolicy.py …`

See SKILL.md § “Distributed training” for caveats.

## Related docs

- `active_adaptation/learning/TEACHME.md` — full style guide (on- + off-policy)
- `scripts/TEACHME.md` — pipeline / run_state notes
- BAC paper code: `lib/Seizing-Serendipity-Exploiting-the-Value-of-Past-Success-in-Off-Policy-Actor-Critic/model/algorithm.py`
