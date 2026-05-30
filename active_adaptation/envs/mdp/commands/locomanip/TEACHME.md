# Loco-Manipulation Command Design

This note describes the current command split used for the relabel-RLPD workflow. The main idea is to first learn a broad loco-manipulation controller with explicit base and end-effector commands, then reuse those rollouts as prior data for a harder end-effector-only task.

## Command Variants

### `SingleEEFLocoManip`

`SingleEEFLocoManip` is the teacher command used by `A2LocoManip`. It exposes two policy-facing command views:

```text
dense:  [v_x, v_y, yaw_rate, eef_x, eef_y, eef_z, pos_diff_x, pos_diff_y, pos_diff_z, eef_pitch, cos(eef_pitch), sin(eef_pitch)]
sparse: [eef_x, eef_y, eef_z, pos_diff_x, pos_diff_y, pos_diff_z]
```

The dense view contains explicit base velocity/yaw commands and an end-effector pitch target. The sparse view keeps only the end-effector position command and current heading-frame EEF-to-target delta. This lets the same rollout contain both the teacher observation and the student-compatible command.

The EEF position convention is shared by both views: `eef_x` and `eef_y` are yaw-aligned offsets from the root, and `eef_z` is height above terrain at the target horizontal location.

`SingleEEFLocoManip` samples a mix of command strategies:

1. Local random commands: sample base velocity, yaw rate, local EEF target, and EEF pitch directly.
2. World-goal commands: sample a persistent world-frame EEF target and a standoff pose, then continuously convert them to local EEF and base commands as the robot moves.

This command is easier to learn because the policy receives direct locomotion guidance while it learns to coordinate the base and arm.

### `LocoManipSparse`

`LocoManipSparse` is the student command used by `A2LocoManipSparse`. It removes the base command and exposes only:

```text
[eef_x, eef_y, eef_z, pos_diff_x, pos_diff_y, pos_diff_z]
```

On reset, it samples a world-frame EEF target near the environment origin and spawns the robot on a ring around that target. At every step it converts the world target into the same heading-frame EEF command convention used by `SingleEEFLocoManip`.

After the EEF reaches the target, the command starts moving the world-frame target continuously. This turns the task from a one-shot reach into a sparse-command tracking problem, while still avoiding explicit base velocity commands.

This task is harder to learn from scratch because the reward only specifies what the EEF should do; the policy must discover the base motion needed to make the target reachable.

## Relabel-RLPD Workflow

The workflow is:

1. Train a teacher policy on `A2LocoManip` with `SingleEEFLocoManip`.
2. Roll out the teacher policy and save trajectories.
3. Relabel the rollout so the student sees the sparse command view and rewards aligned with `LocoManipSparse`.
4. Train SAC on `A2LocoManipSparse` with the relabeled rollout as prior data.

The relabeled prior should remove reward channels that directly supervise base locomotion toward the teacher's dense command. Keep rewards that are compatible with the sparse task, such as EEF position tracking, EEF progress, regularization, survival, and safety/contact terms.

SAC loads the prior archive through `ReplayBuffer.from_rollout`. The prior buffer can compute Monte Carlo returns from the relabeled reward with `ReplayBuffer.compute_return`, storing `ret` and `ret_valid` for diagnostics or prior-value supervision.

## Config Usage

Teacher training:

```bash
python scripts/train.py task=A2LocoManip algo=ppo_symaug
```

Teacher rollout:

```bash
python scripts/rollout.py task=A2LocoManip algo=ppo_symaug checkpoint_path=/path/to/checkpoint.pt
```

Sparse-command SAC training with prior data:

```bash
python scripts/train.py task=A2LocoManipSparse algo=sac algo.prior_data=/path/to/rollout_*.pt
```

Use `scripts/rollout_manager_nicegui.py` to inspect and edit rollout tensors before using them as prior data. In particular, check that the policy observation command matches the sparse command layout expected by `A2LocoManipSparse`.

## Naming

The current names are serviceable, but "dense" and "sparse" can be ambiguous because both commands still contain dense numeric observations. More descriptive alternatives:

- `guided` / `target_only`
- `teacher` / `student`
- `base_eef` / `eef_only`

In code, `dense` and `sparse` are still used as command keys for compatibility.

## Next Step: Object Command

We will completely re-write LocoManipObject. It is no longer intended to be used for training. Instead, it computes scripted commands in the same format as `SingleEEFLocoManip` to manipulate the object (a cloth stand-like object) to move it from one position to another.

It works as follows:

1. Sample initial poses for the object and the robot.
2. Generate appropriate base and EEF commands to approach the object.
3. Generate EEF target commands to enter a grasp pose.
4. Grasp by commanding the gripper to close.
6. Lift the object and move it to the target position.
7. Open the gripper to release the object and backup to the initial pose.

Modify the file accordingly.