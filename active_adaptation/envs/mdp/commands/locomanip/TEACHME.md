# Loco-Manipulation Command Design

This note describes the teacher / student command split used for the
relabel-RLPD workflow. Train a broad loco-manipulation teacher with explicit
base and end-effector commands, then reuse those rollouts as prior data for a
harder **EEF-only** student task.

## Naming: two different “dense / sparse” meanings

1. **Teacher command layout:** `LocoManipNew` exposes a *dense* policy command
   that includes base velocity/yaw **and** EEF targets. The student
   (`LocoManipSparse`) is *EEF-only* (no base command).
2. **Student target density over time:** inside `LocoManipSparse`,
   **goal reaching** uses a **sparse** (persistent) world target, while
   **trajectory following** uses a **dense** sequence of time-varying targets.

Do not confuse (1) with (2).

## Teacher: `LocoManipNew` (`A2LocoManip`)

Three command modes (scheduled from `mode_probs_0` → `mode_probs_1`):

| Mode | Name | Behavior |
|------|------|----------|
| 0 | World | Persistent world-frame EEF goal in a polar annulus about the root (`world_goal_radius_range`, default 1.5–3.0 m) + standoff base pose; loco walks to standoff, EEF tracks the world goal |
| 1 | Body | Body-frame EEF goal that moves with the robot; independent loco command |
| 2 | Nominal | Hold nominal EEF rest pose; independent loco command |

Dense policy command (23D):

```text
[v_x, v_y, yaw_rate, eef_xyz, pos_diff, fwd, fwd_diff, up, up_diff, closed, open]
```

`get_state()` snapshots mode, root/EEF poses, world EEF target, orientation,
gripper status, and `base_pos_error` for offline relabeling. Rollout collection
stores this under `command_state` after each env step.

## Student: `LocoManipSparse` (`A2LocoManipSparse`)

EEF-only policy command (20D):

```text
[eef_xyz, pos_diff, fwd, fwd_diff, up, up_diff, closed, open]
```

Same heading-frame convention as the teacher: `eef_x` / `eef_y` are yaw-aligned
offsets from the root; `eef_z` is absolute height. No base velocity or yaw rate.

### Mode 0 — Goal reaching (sparse target)

- Spawn near the env origin (`goal_spawn_radius_range`, small jitter).
- Sample / update world goals with the **same Warp helpers** as
  `LocoManipNew` mode 0 (`sample_world_goal` / `update_world_command` in
  `loco_manip_kernels.py`): polar annulus (`world_goal_radius_range`), standoff,
  heading-frame EEF refresh, and `base_pos_error` for reward gates.
- Policy observation stays EEF-only (no base velocity command).
- After the EEF is close enough, commands may be resampled.

### Mode 1 — Trajectory following (dense targets)

- Sample a simple parametric curve in world frame (circle or line segment, with
  optional small height oscillation).
- Advance the curve phase each step (`ω · dt`) so the EEF target moves continuously.
- Spawn closer to the curve center (`traj_spawn_radius_range`).

Online mix is controlled by `trajectory_prob` (default `0.5`).

This task is harder to learn from scratch than the teacher: the reward only
specifies what the EEF should do; base motion is implicit.

## Relabel-RLPD Workflow

```text
1. Train teacher PPO on A2LocoManip (LocoManipNew)
2. Roll out teacher → archive with command_state
3. Relabel with A2LocoManipSparse (LocoManipSparse.relabel_command + reward relabel)
4. Train SAC / RLPD on A2LocoManipSparse with the relabeled prior
```

### Relabel mapping (`LocoManipSparse.relabel_command`)

| Teacher mode | Student mode | Target at step `t` |
|--------------|--------------|--------------------|
| 0 (world) | Goal reaching | Teacher world EEF goal (`cmd_eef_pos_w`), teacher `cmd_eef_rot_w` |
| 1 or 2 (body / nominal) | Trajectory following | Achieved `eef_pos_w[t+1]` and `eef_quat_w[t+1]` (hindsight); on `done`, fall back to step `t` |

Both paths rebuild the EEF-only `command` / `next.command` and the tracking
fields used by reward `relabel()` (`pos_error_*`, `forward_diff_w`,
`upward_diff_w`). Teacher `base_pos_error` is kept for tracking reward gates.

After relabel + RLPD, the policy only needs an EEF target pose or a dense
target sequence; base movement stays implicit.

**Expectation:** online training from scratch on `A2LocoManipSparse` is much
harder than bootstrapping from relabeled teacher data via RLPD.

Drop teacher reward channels that supervise base locomotion toward the dense
command (`linvel_exp`, `angvel_z_exp`, …). Keep sparse-compatible terms: EEF
position / orientation tracking, grasp, regularization, survival, safety.

SAC loads the prior through `ReplayBuffer.from_rollout`. Optional Monte Carlo
returns: `ReplayBuffer.compute_return` → `ret` / `ret_valid`.

## Config Usage

Teacher training:

```bash
python scripts/train_ppo.py task=A2/A2LocoManip algo=ppo_symaug
```

Teacher rollout (mixed modes so both relabel paths appear):

```bash
python scripts/rollout.py task=A2/A2LocoManip algo=ppo_symaug \
  checkpoint_path=/path/to/checkpoint.pt
```

Relabel:

```bash
python scripts/relabel.py task=A2/A2LocoManipSparse \
  rollout_path=/path/to/rollout_*.pt
```

Sparse SAC with prior:

```bash
python scripts/train_offpolicy.py task=A2/A2LocoManipSparse algo=sac \
  algo.prior_data=/path/to/rollout_*.relabeled.pt
```

Or chain stages with `scripts/pipeline.py recipe=a2_relabel_rlpd`.

Use `scripts/rollout_manager_nicegui.py` to inspect archives. After automatic
relabel, `command` should already match the sparse layout.

## Related files

- Teacher: `loco_manip_new.py`
- Student: `loco_manip_sparse.py`
- Tasks: `cfg/task/A2/A2LocoManip.yaml`, `cfg/task/A2/A2LocoManipSparse.yaml`
- Entry points: `scripts/rollout.py`, `scripts/relabel.py`, `scripts/pipeline.py`
