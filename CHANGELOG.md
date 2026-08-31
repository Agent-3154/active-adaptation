# Changelog

All notable changes to **active-adaptation** are documented here.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers match `pyproject.toml`. Git tags and the **`v0.8`** branch mark supported release lines; **`main`** at `0.7.x` is the pre-release trunk this document compares against.

---

## [Unreleased]

### Added

- **`task.nan_guard`** — controls non-finite transition handling in `_EnvBase`:
  - `sanitize`: zero invalid rows and terminate those envs (existing behavior)
  - `error`: raise `NonFiniteTransitionError` with offending `env_ids` and tensor keys
  - `off`: skip the guard entirely
  - default: `error`; tasks that want bad-env isolation must opt into `sanitize`
  - Override via env var `AA_NAN_GUARD`.

### Deprecated

- **`reward._mult_dt_` / `env.mult_dt`** — multiplying per-step rewards by `step_dt` in `_compute_reward` is deprecated and will be removed. Set `reward._mult_dt_: false` in task config and scale rewards in the policy / algorithm instead.

---

## [0.8.1] — 2026-08-30

First consolidated **v0.8** release line. This branch merges months of HDMI / MimicLite parity work, deferred MDP construction, mjlab hardening, and project-packaging improvements that were not on `main` (`0.7.0`).

**Upgrade from 0.7.x:** check out branch `v0.8`, run `uv sync` plus `uv sync --project venv/isaac51` (and/or `venv/mjlab`), refresh robot assets under `.cache/aa-robot-models/`, then run `aa-project discover`. Extension projects that define custom MDP terms will need code updates — see **Breaking changes** below.

### Environment & simulation

- **Deferred MDP lifecycle** — observation, reward, termination, action, command, and randomization terms are constructed from Hydra kwargs first, then bound in `_initialize(env)` after the scene exists. Terms no longer take `env` in `__init__`.
- **Command step order** — `prescribe` runs before action processing (command-driven inputs such as `arm_control`); `sync_state` refreshes intermediates for rewards; `update` writes next-step targets for observations. Supports split policy / reference control (HDMI, manipulation tracking).
- **`env.episode_origin`** — commands set per-episode spawn origins (separate from terrain layout slots) for shared 3DGS and episode-local frames.
- **`reset(env_ids, tensordict)`** — all MDP `reset` hooks use this signature; terms may read/write the reset tensordict.
- **Scene-owned sensors** — contact and camera sensors are declared in task YAML (`sensors:`) and built via factories under `envs/sensors/`, with backend-specific contact data handled explicitly (Isaac `net_forces_w` / `force_matrix_w` vs mjlab `force`).
- **mjlab 1.6** — primary MuJoCo backend for new work; collision/sensor cfg aligned with mjlab ≥1.6 structural fields. Standalone `backend=mujoco` remains in tree but is deprecated.
- **Isaac Viser viewer** — optional browser viewer (`viewer.viser: true`) alongside Kit; debug drawing goes through `env.scene.draw_*` / camera frustums, not legacy `env.debug_draw`.
- **3D Gaussian splatting & mesh composite** — optional `task.visual` stack (`FvdbGaussianWorld`, robot mesh overlay via `simple-raycaster` / nvdiffrast) for appearance-rich tasks.
- **Non-finite rollout guard** — invalid transition rows are zeroed and those envs are terminated so one bad sim instance does not poison PPO updates.
- **Functional observation groups** — an obs group may be all “functional” (nested TensorDict per term) or all dense (concatenated vector); mixing within a group is rejected.

### MDP & tasks

- **Trajectory tracking command** (`TrajTracking`) for time-indexed reference motion.
- **End-effector pose actions** — `EndEffectorPose` / `EndEffectorPoseDelta` with damped least-squares IK (`utils/ik.py`).
- **Exteroceptive obs refactor** — `raycast_camera`, height map, and related terms moved under `envs/mdp/observations/extero/` with improved normal handling for mesh raycasts.
- **Visual observations** — `gs_camera` and related terms tied to `env.visual` / episode origins.
- **Locomotion / tracking rewards** — EEF pos/vel tracking, body angular-velocity penalty, grouped reward diagnostics kept finite under bad env isolation.
- **Terminations** — max episode length as a termination class; `bodies_too_close`, segment-crossing checks, and related safety terms expanded.
- **Underwater** — BlueROV Heavy + arm asset/config updates; ocean / propulsion terms adjusted for deferred init.
- **Primitive rigid objects** — lightweight spawnable props via `assets/primitive_objects.py`.

### Assets

- **`AssetSpec` pattern** enforced for articulated robots — shared `joint_names_simulation` / `body_names_simulation`, symmetry maps, and matching Isaac + mjlab contact sensors.
- **Composed robots** — e.g. Unitree A2 + Piper arm (`a2_manipulator`); MJCF built with sibling [**assetx**](../aa-projects/assetx/) recipes, published to `.cache/aa-robot-models/`.
- **Cross-backend indexing** — MDP code should use `find_joints` / `find_bodies` / `find_sensor_bodies`, not backend-native `asset.find_*` order.
- **Agent skill** — `.agents/skills/asset-definition/` documents the assetx → cache → factory pipeline.

### Training & algorithms

- **HDMI / MimicLite path** — `train_imitation.py` reworked; frame-invariant correction inputs, motion-tracking commands, and parity fixes carried from the release branch.
- **PPO hardening** — non-finite gradient backstop, isolated simulator failure handling, grouped-reward / GAE alignment, legacy checkpoint resume fixes.
- **Teacher–student & imitation extras** — EMA teacher, DAgger improvements, SAC behavioral-cloning loss hook, ASPO / interprior-style post-training experiments.
- **Symmetry augmentation** — per-observation importance weights for `ppo_symaug`; consistent TensorDict key ordering across backends.
- **Policy construction** — `make_env_policy` / algo cfg instantiation pattern updated; creating policies from checkpoints improved.
- **WandB utilities** — richer run querying and diagnostics helpers in `utils/wandb.py`.
- **Launch** — `launch_ddp.sh` updates and new `launch_multinode_tmux.sh` for multi-node tmux workflows.

### Projects & tooling

- **Unified project CLI** — `aa-project` (Typer) replaces scattered entry points (`aa-create-project`, `aa-pull`, `aa-recent-commands`, …). Subcommands: `create`, `install`, `discover`, `enable` / `disable`, `pull`.
- **Entry-point discovery** — extension projects register via `active_adaptation.projects` and `active_adaptation.learning`; manifest at `.cache/projects.json`.
- **uv multi-environment workflow** — backend stacks isolated in `venv/isaac51` and `venv/mjlab`; root env is tooling-only. Optional `[render]` extra for 3DGS / nvdiffrast.
- **Torch 2.11** pin and dependency cleanup (removed unused `moviepy`, `av`, `pygame`, root-level `mujoco` pin).
- **First unit tests** — non-finite transition sanitization and reward EMA finiteness.

### Breaking changes (0.7.x → 0.8.x)

| Area | Before (0.7.x) | After (0.8.x) |
|------|----------------|---------------|
| MDP term `__init__` | `def __init__(self, env, …)` | `def __init__(self, …)` + `_initialize(self, env)` |
| Class names | `*V2` suffix in places | Unified names (`Reward`, `Observation`, …) |
| Debug draw | `env.debug_draw.*` | `env.scene.draw_*`, `create_camera_frustum` |
| Projects | ad-hoc scripts / path hacks | Packaging + `aa-project discover` / `enable` |
| MuJoCo backend | `backend=mujoco` common | Prefer `backend=mjlab` |
| Robot models | In-repo / hand-edited MJCF | `ROBOT_MODEL_DIR` + assetx for composed variants |
| Contact forces | Assumed shared sensor API | Branch on backend or use task `sensors:` factories |
| CLI | `aa-create-project`, `aa-pull`, … | `aa-project …` |

The first observation after reset is intentionally discarded in training (`is_init` mask); do not recompute command targets in `reset` to “fix” it.

### Known limitations

- **`venv/isaac60`** (Isaac Lab 3 / Sim 6) is tracked on branch **`v0.9`**, not validated on `v0.8`.
- **`main`** is not kept in sync with `v0.8`; new work should branch from **`v0.8`** (or later release branches).
- Some legacy config paths (policy class as Hydra `_target_`) still work via fallbacks but should migrate to algo cfg objects.

---

## [0.7.0] and earlier

Pre-`v0.8` history lives on branch **`main`**. No curated changelog was maintained before this file; use `git log main` for commit-level detail.
