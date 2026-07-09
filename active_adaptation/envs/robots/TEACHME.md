# Underwater Robot Design Notes

This document captures the current BlueROV underwater dynamics implementation in
`active_adaptation/envs/robots/underwater.py` and how it is wired into the
asset/backend stack.

## Terminology

- **Thruster**: physical propulsion unit (Blue Robotics T200-like device).
- **Rotor**: simulation body/joint name used by USD and articulation APIs
  (`rotor_0`, `rotor_1`, ...). In code we keep this naming to stay aligned with
  body lookup and tensor indexing.
- **Throttle command**: normalized action-level command in `[-1, 1]`.
- **Wrench**: 6D force/torque vector `[Fx, Fy, Fz, Mx, My, Mz]` in body frame.

In practice, "thruster" and "rotor" refer to the same actuated channel. We use
"rotor" for IDs/names and "thruster" for physical interpretation.

## Hydrodynamics Model

Per environment, we compute body-frame hydrodynamic terms each pre-step:

- Relative body velocity is computed by subtracting sampled flow/current from
  base link velocity (with frame/sign conventions handled explicitly).
- Body acceleration is estimated from finite differences and low-pass filtered
  with `acc_filter_alpha`.
- Damping uses linear + quadratic terms:
  `D(v) = D_lin + D_quad * |v|` (implemented with a maintained 6x6 form).
- Added mass: `M_a * a`.
- Coriolis-like term from added-mass momentum.
- Buoyancy from displaced volume, gravity, and center-of-buoyancy offset.

Final hydro wrench is:

- `hydro = -(added_mass + coriolis + damping)`
- Total base wrench = `hydro + buoyancy`

The wrapper stores decomposed terms (damping, added mass, coriolis, buoyancy,
hydro) in `UnderwaterRobotData` for debugging/analysis.

## Actuator (Thruster) Model

Thruster commands are converted to forces in three stages:

1. **Command filter**: `throttle_cmd` is clamped to `[-1, 1]`, then first-order
   filtered using per-rotor time constants.
2. **Throttle -> RPM map**: piecewise affine mapping with deadzone around zero,
   then clamped to max RPM.
3. **RPM -> thrust map**: piecewise quadratic fit with sign-aware coefficients,
   scaled by force constants.

Generated thrust is applied along local rotor x-axis as body-frame force for
each rotor body.

## Wrapper Lifecycle and Integration

Current design uses **instance-based wrappers**:

- Asset declarations return `AssetSpec(config=..., sensors=..., wrapper=...)`.
- The backend (`IsaacBackendEnv.setup_scene`) receives `asset_spec.wrapper`,
  calls `wrapper._initialize(robot=self.robot, env=self)`, then registers
  optional lifecycle callbacks (`startup`, `reset`, `pre_step`, `post_step`,
  `update`, `debug_draw`).

`UnderwaterRobot.__init__` now keeps only config/state setup that does not
depend on parsed robot assets. Asset parsing and tensor allocation are done in
`_initialize(...)`.

## Stepping Logic

- `reset(...)`: clears velocity/acceleration history and resamples flow
  disturbance for selected envs.
- `pre_step(...)`: calls `write_data_to_sim()`.
- `write_data_to_sim()`:
  - reads articulation root state,
  - computes hydrodynamic + buoyancy terms,
  - computes rotor thrust from throttle state,
  - writes external forces/torques via
    `robot.set_external_force_and_torque(...)` on base + rotor bodies.
- `debug_draw()`: visualizes rotor thrust vectors when GUI/debug draw is active.

## Why This Split

- Keeping `AssetSpec.wrapper` as an instance avoids backend-specific constructor
  signatures in each asset file.
- Deferring heavy setup to `_initialize(...)` ensures wrapper allocation is
  consistent with final robot instance/device/num_envs.
- Callback registration keeps the wrapper backend-agnostic while fitting the
  environment's existing lifecycle hooks.