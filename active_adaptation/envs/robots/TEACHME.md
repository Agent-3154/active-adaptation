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

- Relative body velocity is computed by subtracting sampled flow/current
  (with frame/sign conventions handled explicitly).
- Body acceleration (base) is estimated from finite differences and low-pass
  filtered with `acc_filter_alpha`.
- **Per-body damping** on each link's flow-relative 6D twist:
  `τ = (D_lin + D_quad ∘ |v|) v` (diagonal). Specs are lists or
  `{name_regex: value}` maps; each value is a `float` (isotropic translation),
  length-3 (anisotropic translation), or length-6 (full wrench). Every body
  must be specified.
- Added mass: `M_a * a` (base only).
- Coriolis-like term from added-mass momentum (base only).
- Buoyancy from **per-body** displaced volume, gravity, and center-of-buoyancy
  offset (`coBM` on the base body only).

Final wrenches:

- Every body: buoyancy − damping
- Base additionally: `-(added_mass + coriolis)`
- Rotors additionally: thruster forces

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
- Both Isaac and mjlab backends receive `asset_spec.wrapper`, call
  `wrapper._initialize(robot=self.robot, env=self)`, then register optional
  lifecycle callbacks (`startup`, `reset`, `pre_step`, `post_step`, `update`,
  `debug_draw`).

`UnderwaterRobot.__init__` keeps only config/state that does not depend on the
parsed robot. Asset parsing and tensor allocation happen in `_initialize(...)`.

## Stepping Logic

- `reset(...)`: clears velocity/acceleration history and resamples flow
  disturbance for selected envs.
- `pre_step(...)`: calls `write_data_to_sim()`.
- `write_data_to_sim()`:
  - reads articulation root / body state,
  - computes hydrodynamic + buoyancy terms,
  - computes rotor thrust from throttle state,
  - writes body wrenches: Isaac via `permanent_wrench_composer`
    (`is_global=False`); mjlab via world-frame `write_external_wrench_to_sim`.
- `debug_draw()`: visualizes rotor thrust vectors when GUI/debug draw is active.

## Why This Split

- Keeping `AssetSpec.wrapper` as an instance avoids backend-specific constructor
  signatures in each asset file.
- Deferring heavy setup to `_initialize(...)` ensures wrapper allocation is
  consistent with final robot instance/device/num_envs.
- Callback registration keeps the wrapper backend-agnostic while fitting the
  environment's existing lifecycle hooks.

## Real2Sim / System Identification

Goal: fit the **parameters of this model** from tank / pool / sea data so sim
matches the vehicle you care about — not invent a richer dynamics law. If a
phenomenon is outside the model (see gaps below), either accept the residual or
extend the code first.

### What this model can represent

| Term | Scope | Tunable knobs (asset / `HydrodynamicsCfg`) |
|------|--------|---------------------------------------------|
| Rigid mass / inertia | Sim body inertials (USD/MJCF) | Keep geometry mass as dry mass; do **not** bake Ma into inertials |
| Buoyancy | Per-body | `volume` (every body), `coBM` (base trim arm only), `water_density`, `gravity` |
| Linear / quadratic drag | Per-body diagonal | `linear_damping`, `quadratic_damping` (`float` / len-3 / len-6) |
| Added mass + Coriolis | **Base twist only** | `added_mass` diagonal 6-tuple; Coriolis is derived, not free |
| Acc estimate | Base Ma path | `acc_filter_alpha` (sim-side; usually leave alone) |
| Thrusters | Per rotor body | `rotor_time_constants`, `rotor_force_constants`; RPM/thrust maps are **hardcoded** in `write_data_to_sim` |
| Flow | Env disturbance | `set_flow_velocities` (not vehicle ID) |

Hardcoded thruster maps (throttle→RPM, RPM→thrust) came from T200-style fits.
For a new thruster family, prefer editing those maps (or making them cfg) before
over-tuning `force_constants` alone.

### Sign / frame convention (do not forget when fitting)

Hydro coeffs in this stack use a **MarineGym-style axis flip** on indices
`[1, 2, 4, 5]` relative to Isaac/mjlab body frames. Fitted numbers must use the
**same convention as the asset YAML** (e.g. BlueROVHeavy MarineGym dumps). If you
fit in NED / ROS body frames, convert before writing into `LINEAR_DAMPING` /
`ADDED_MASS`.

### Recommended ID order (isolate terms)

Identify **statics → actuators → vehicle drag → Ma → arm**, in that order. Joint
optimization of everything against free flight is badly conditioned.

1. **Geometry / mass**
   - Confirm dry mass, CoM, and inertias against CAD or hanging tests.
   - Align sim body tree (especially arm) with the real vehicle.

2. **Static buoyancy / trim** (zero throttle, still water)
   - Measure float depth / net heave force and pitch–roll equilibrium.
   - Fit `volume["base_link"]` (and arm volumes if they change trim with pose)
     so net weight ≈ buoyancy at the operating depth.
   - Fit `coBM` from pitch/roll restoring moment (small lean tests).
   - Prefer mesh volumes (`assetx` volume tool) as priors; scale globally if the
     vehicle is slightly positive/negative buoyant.

3. **Thrusters** (docked / locked vehicle or known load cell)
   - Step throttle and record force along each thruster axis.
   - Fit per-rotor `force_constants` (and `time_constants` from rise time).
   - If forward/reverse asymmetry disagrees with the hardcoded RPM→thrust
     polynomials, update those maps — do not absorb all error into drag.

4. **Vehicle damping (base)** — coast / constant-velocity runs
   - Hold attitude with gyros or cage; command steady surge / sway / heave /
     yaw one axis at a time when possible.
   - At low speed, linear terms dominate; at higher speed, quadratic.
   - Fit `LINEAR_DAMPING["base_link"]` and `QUADRATIC_DAMPING["base_link"]`
     (full length-6 when moments matter).
   - Rotors: leave damping `0` (dummy bodies).

5. **Added mass (base)** — acceleration / forced oscillation
   - Needs good acceleration (IMU fused or motion capture).
   - Excite one DoF; fit diagonal `ADDED_MASS`. Off-diagonals are **not** in the
     model — residual coupling will look like drag or Coriolis error.
   - Coriolis is fully determined by `ADDED_MASS` and twist; do not treat it as
     an independent parameter.

6. **Arm / manipulator hydro** (HeavyArm and similar)
   - Base Ma stays vehicle-level; arm uses **per-link volume + damping only**.
   - Fix volumes from meshes first.
   - Fit arm damping with the vehicle locked or at rest: move one joint at a
     time (or hold poses in flow) and match joint torque / coast-down.
   - Start with isotropic `float` translational drag; promote to len-3/len-6
     only if data supports it.
   - Placeholder arm coeffs in assets are order-of-magnitude only until fitted.

### Practical experiment checklist

- Log at ≥ control rate: root twist, body quats, thruster commands, (optional)
  IMU accel, joint q/τ for arms.
- Prefer still water (`max_flow_vel = 0`) for ID; use flow randomization later
  for domain randomization, not for fitting nominal coeffs.
- Replay logged throttle into sim (`throttle_cmd`) and compare open-loop
  trajectories; iterate on the stage above that explains the residual.
- Inspect `robot.data_underwater` terms (`damping`, `added_mass`, `coriolis`,
  `buoyancy`, `thrusts_b`) to see which wrench is wrong before retuning
  everything.

### Where to put fitted values

- Vehicle constants live on the asset module
  (`assets/underwater/BlueROV*.py`: `VOLUME`, `COBM`, `ADDED_MASS`,
  `LINEAR_DAMPING`, `QUADRATIC_DAMPING`, `ROTOR_*`).
- Keep Isaac and mjlab factories sharing the same dicts so backends stay aligned.
- Document the data source / date in a short comment next to the constants.

### Known model gaps (sysid will not fix these)

- No per-link added mass / arm–vehicle hydrodynamic coupling.
- Diagonal Ma and diagonal damping only (no full 6×6 matrices).
- Thruster–thruster / thruster–hull interaction not modeled (allocation assumes
  independent body-local +X forces).
- No surface effects, tether drag, or compressible free-surface terms.
- Rotor collision/visual bodies are kinematic placeholders for force application.

If closed-loop tracking still diverges after the staged fits above, the residual
is likely one of these gaps — extend `UnderwaterRobot` rather than dumping the
error into `QUADRATIC_DAMPING`.