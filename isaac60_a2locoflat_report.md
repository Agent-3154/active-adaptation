# Isaac Sim 6 / Lab 3 A2LocoFlat: training progress and rollout-speed vs Isaac 5.1

**Date:** 2026-08-19  
**Repo:** `active-adaptation` branch `v0.9` (HEAD `16f66eb8`)  
**Task:** `A2/A2LocoFlat`, `algo=ppo`, `algo.in_keys=[policy,extero]`, headless, wandb disabled  
**Hardware:** 2× RTX 5090 (isaac60 on GPU 0, isaac51 on GPU 1); other GPUs idle except GPU 7  

Stacks:

| Name | Python | Isaac Sim | Isaac Lab | Warp | Venv |
|------|--------|-----------|-----------|------|------|
| **isaac60** | 3.12 | 6.0.0.1 | Lab 3 (`IsaacLab60`, `isaaclab` 6.1.11 / `perf-2026-06-24`) | 1.16.0 | `venv/isaac60` |
| **isaac51** | 3.11 | 5.1.0 | Lab 2.3.2 (`IsaacLab/`) | &lt; 1.13 | `venv/isaac51` |

Newton is out of scope. Sibling `IsaacLab/` was not modified.

---

## 1. Result in one paragraph

A2LocoFlat on **isaac60 now trains**. Critic `value_mean` rose from ~0.02 to ~0.32 over four PPO iterations at 4096 envs (same direction as isaac51). **Rollout FPS at 4096 envs is lower on isaac60 than isaac51:** about **51k vs 76k** after the first (warmup) iteration — isaac51 is **~1.48×** faster. First-iteration FPS is 57k (isaac60) vs 78k (isaac51).

---

## 2. Training progress (`critic/value_mean`)

PPO `train_every=32`. Frames per iteration = `num_envs × 32`.

### isaac60, 1024 envs (smoke, 3 iters, `total_frames=98304`)

| Iter | env_frames | value_mean | rollout_fps |
|------|------------|------------|-------------|
| 0 | 32768 | 0.021 | 25411 |
| 1 | 65536 | 0.118 | 28795 |
| 2 | 98304 | 0.218 | 28192 |

### isaac60, 4096 envs (4 iters, `total_frames=524288`)

| Iter | env_frames | value_mean | rollout_fps | rollout_s |
|------|------------|------------|-------------|-----------|
| 0 | 131072 | 0.019 | 57307 | 2.29 |
| 1 | 262144 | 0.119 | 52967 | 2.47 |
| 2 | 393216 | 0.217 | 50618 | 2.59 |
| 3 | 524288 | 0.318 | 50786 | 2.58 |

Post-warmup FPS: **min 50618 / mean 51457 / max 52967**.

### isaac51, 4096 envs (same Hydra overrides)

| Iter | env_frames | value_mean | rollout_fps | rollout_s |
|------|------------|------------|-------------|-----------|
| 0 | 131072 | −0.003 | 77619 | 1.69 |
| 1 | 262144 | 0.071 | 72560 | 1.81 |
| 2 | 393216 | 0.130 | 77637 | 1.69 |
| 3 | 524288 | 0.201 | 78900 | 1.66 |

Post-warmup FPS: **min 72560 / mean 76366 / max 78900**.

`value_mean` is return-target mean, not episode return. Four iterations are enough to show the critic is moving, not to compare final loco quality. Both stacks terminate early episodes via `fall_over` (untrained policy).

---

## 3. Rollout-speed comparison (4096 envs)

`performance/rollout_fps` = `num_envs × train_every / rollout_time` (collector only, not PPO update).

| | isaac60 (Sim 6 / Lab 3) | isaac51 (Sim 5.1 / Lab 2) | Ratio 51/60 |
|--|-------------------------|---------------------------|-------------|
| First iter FPS | 57.3k | 77.6k | 1.35× |
| Post-warmup mean FPS | **51.5k** | **76.4k** | **1.48×** |
| Last iter FPS | 50.8k | 78.9k | 1.55× |
| Last iter rollout time | 2.58 s | 1.66 s | 0.64× |
| Wall clock (Kit + scene + 4 iters) | ~62 s (05:26:43–05:27:45) | ~32 s (05:37:12–05:37:44) | — |

isaac60 at **1024** envs was ~28k FPS (post-warmup). Scaling 1024→4096 on isaac60 is about **1.8×** FPS, not 4× — typical for this stack once CPU/Kit overhead is no longer the only term.

Logs:

- `active-adaptation/outputs_train/isaac60_A2LocoFlat_train.log` (1024 smoke)
- `active-adaptation/outputs_train/isaac60_A2LocoFlat_4096.log`
- `active-adaptation/outputs_train/isaac51_A2LocoFlat_4096.log`

---

## 4. Patches that unblocked isaac60 (this session)

Lab 3 façade work already on `v0.9` before this loop: WXYZ adapter, torch 2.11 pin, seed without `isaacsim.core.utils.torch`, duck-typed scene/sim APIs.

Then, in order of crashes:

1. **`simple_raycaster` import on tasks without 3DGS** (`373075aa`)  
   `envs/visual/__init__.py` imported `fvdb_gs` → `simple_raycaster`. That package is **not** in isaac60 (`warp-lang==1.16` vs `simple-raycaster`’s `warp-lang<1.13`). Skip visual-world import when `task.visual` is unset. Ground-only `height_scan` now uses Isaac Lab `raycast_mesh` instead of `MultiMeshRaycaster`.

2. **Body-mass randomization on GPU ProxyArray** (`159cdc53`)  
   Lab 2 did `scale.cpu()` for PhysX views. Lab 3 `default_mass` is GPU. Write via `set_masses_index` / `set_inertias_index` when present.

3. **Joint writers on the WXYZ façade** (`b01fd860`, `f2abebb2`, `1086ee00`)  
   Lab 3 `_index` kernels want **int32** env/joint ids and keyword payloads. Wrap `write_joint_*` and `set_joint_*_target` on `CanonicalIsaacAsset`.

4. **Material randomization Warp vs torch** (`be9df835`, `16f66eb8`)  
   Lab 3 `get_material_properties()` is a Warp array. Lab 2 `set_material_properties` wants a flattened torch buffer (passing Warp arrays fails inside Warp, not with `TypeError`). Branch on `isaaclab_uses_xyzw()`.

5. **Contact `compute_first_contact` dtype** (`2cca92b0`)  
   Lab 3 returns a float `ProxyArray`. Gait rewards use `torch.where`. Convert to `bool` on `CanonicalIsaacSensor`.

6. **Shared `/tmp/isaaclab/logs`** (`a6c6411e`)  
   Lab 2 `SimulationContext` logs to `/tmp/isaaclab/logs`, owned here by another user. Set `SimulationCfg.log_dir` to `~/.cache/isaaclab/logs`.

---

## 5. Issues that remain

### Blocking for other tasks (not A2LocoFlat)

- **`simple-raycaster` is not installable on isaac60** as published (`warp-lang<1.13`). Tasks that need `MultiMeshRaycasterV2`, mesh RGB-D / 3DGS composite, or USD mesh extraction via `simple_raycaster.utils_usd` will fail until that pin is lifted or the package is vendored against Warp 1.16.
- Leftover **lazy** `isaacsim.core.utils.stage` imports in `meshes.py`, `extero.py` (`targets=` path), `viewer.py` — not hit by this loco task.
- **`nvdiffrast` / `[render]` extra** still omitted from isaac60 by design (CUDA extension build).

### isaac60 runtime noise (non-fatal)

- **PhysX contact-report warnings** for every env on `/World/envs/env_*/Robot/base_link/visuals/base_link` (“Failed to find contact report API”). Floods the 4096-env log (~12k lines). Contact terms still ran (`feet_air_time`, `undesired_contact`, crash). Looks like contact sensors matching **visual** prims. Worth tightening `ContactSensorCfg` filters; not investigated further here.
- PhysX TGS warning: `enable_external_forces_every_iteration=False`.
- Kit: no display / MaterialX / `isaaclab_visualizers` missing `extension.toml` on this Lab 3 checkout.
- `resource_tracker` leaked-semaphore warning on process exit (also on isaac51).

### Lab 3 API debt inside AA

- Many MDP terms still call Lab 2-shaped APIs; the façade converts. Not all `write_*` / `set_*` / PhysX view methods are wrapped (armature, friction coefficient, CoM, `root_physx_view` consumers besides materials).
- `asset.data.body_materials = …` on Lab 3 writes onto `CanonicalData`, not the PhysX buffer (write path goes through `root_view`; the attribute set is a leftover cache).
- `height_scan` with `targets=` still needs `simple_raycaster` and `isaacsim.core.utils.stage`.
- Contact force field names / `ProxyArray` indexing may still surprise terms that assume Lab 2 torch bools.

### isaac51 / host

- `/tmp/isaaclab/logs` is not world-writable on this machine. AA now bypasses it; other Lab 2 tools that log there will still fail for this user.
- isaac51 comparison used the **same `v0.9` tree** (including Lab 3-safe height_scan). That is what we want for a fair backend compare.

### What was not measured

- Longer training (policy quality, `train/stats/loco/*` curves).
- Rough terrain, cameras, locomanip, mjlab.
- Newton / OVPhysX backends inside Lab 3 (`randomize_rigid_body_material` is a no-op on OVPhysX per Lab 3 events.py).

---

## 6. How to reproduce

```bash
# isaac60, 4096 envs
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
  uv run --project venv/isaac60 python scripts/train_ppo.py \
  task=A2/A2LocoFlat algo=ppo 'algo.in_keys=[policy,extero]' \
  wandb.mode=disabled eval=false task.num_envs=4096 \
  total_frames=524288 headless=true

# isaac51, 4096 envs
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
  uv run --project venv/isaac51 python scripts/train_ppo.py \
  task=A2/A2LocoFlat algo=ppo 'algo.in_keys=[policy,extero]' \
  wandb.mode=disabled eval=false task.num_envs=4096 \
  total_frames=524288 headless=true
```

PPO default `in_keys` include a `command` group; this task’s groups are `policy` and `extero` (`command` is a term inside `policy`). The override is required on both stacks.

Robot USD: `.cache/aa-robot-models/a2/` (Hugging Face `btx0424/aa-robot-models`).
