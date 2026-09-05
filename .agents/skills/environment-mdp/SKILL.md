---
name: environment-mdp
description: Implement and wire MDP terms in active-adaptation (observations, rewards, terminations, actions, commands, randomizations) using deferred-init bases. Use when adding or modifying classes under envs/mdp/, editing cfg/task/ observation/reward/termination/input/command/randomization/sensors blocks, implementing sealed Command/Reward update→_update with in_keys/out_keys, command prescribe/step for split policy/command control, registering scene sensors (envs/sensors/, contact_sensor), debugging env_base step/reset callbacks, cross-backend contact sensor data (Isaac net_forces_w / force_matrix_w vs mjlab force), debug viz (scene.draw_*, camera frustums), Warp/simple-raycaster raycasting, USD mesh extraction, or Isaac/mjlab Viser viewers.
---

# Environment / MDP terms (active-adaptation)

Implement MDP terms with **deferred init**: construct from Hydra kwargs **without** an env, then bind in `_initialize(env)` after the scene exists. Register via `RegistryMixin`; task YAML selects them by class name. `Command` / `Reward` use a sealed `update` → `_update` dispatcher with optional `in_keys` / `out_keys`.

**Primary references**
- Env wiring + step/reset: `active_adaptation/envs/env_base.py`
- Shared hooks: `active_adaptation/envs/mdp/base.py` (`MDPComponent`)
- Term bases: `envs/mdp/{observations,rewards,terminations,actions,commands,randomizations}/base.py`
- Scene sensors: `active_adaptation/envs/sensors/` (registry group `"sensor"`); backends merge into `scene.sensors`
- Name / index helpers: `active_adaptation/envs/utils/api.py` (`find_joints`, `find_bodies`, `find_sensor_bodies`)
- Task configs: `cfg/task/**/*.yaml`
- Debug viz API: `active_adaptation/envs/adapters.py` (`SceneAdapter`, `CameraFrustumHandle`)
- Raycasting / USD meshes: installed package `simple-raycaster` (see section below + [reference.md](reference.md))

Read [reference.md](reference.md) for the step-loop diagram, callback registration, file map, scene-sensor wiring, viz backends, name-order / contact-index rules, and raycast/mesh details.

**Related skills:** `onpolicy-algorithms`, `offpolicy-algorithms`, `asset-definition`.

---

## When to use

- Adding a new observation / reward / termination / action / command / randomization
- Porting a term to deferred init / sealed `_update`
- Wiring or renaming terms in `cfg/task/`
- Splitting policy vs command actuation (`input.action` + `command.prescribe` → `input.arm_control`, etc.)
- Declaring task-level scene sensors (`sensors:` — object contact, filtered ground contact, …)
- Debugging missing callbacks, wrong step order, or registry lookup failures
- Debug vectors / points / plots / camera frustums across Isaac and mjlab
- Isaac multi-mesh raycasting, USD → trimesh/Warp extraction, or browser Viser robot meshes

---

## Hard rules

1. **Use deferred bases only** — `Observation`, `Reward`, `Termination`, `Action`, `Command`, `Randomization`. Construct from Hydra kwargs without `env`; bind in `_initialize(env)`.
2. **Two-phase init** — `__init__(**cfg_kwargs)` stores config only; `_initialize(env)` binds `self.env`, caches assets/sensors, allocates buffers. Always call `super()._initialize(env)` first when overriding.
3. **Register by subclassing** — subclassing a base auto-registers under the class name (or `namespace.ClassName`). Import the module so registration runs (see package `__init__.py` auto-import patterns).
4. **Shape convention** — tensors are batched `(num_envs, …)`. Rewards/terminations usually return `(num_envs, 1)`.
5. **Sealed `Command.update` / `Reward.update`** — do **not** override `update`. Implement `_update(*tensors_in)` instead. Optional `in_keys` / `out_keys` class attrs; missing keys are passed as `None`. `_update` parameters **must not** have defaults (enforced by `check_update_signature`). `startup` / `reset` are called explicitly from `_EnvBase` on command, adaptations, obs/reward groups, randomizations, terminations, and input managers. `_add_mdp_component` still registers overridden `pre_step` / `post_step` / `debug_draw` via `is_method_implemented`.
6. **Simulation name order** — Isaac and MuJoCo/mjlab typically use different joint/body orders. Resolve names via `asset.cfg.joint_names_simulation` / `asset.cfg.body_names_simulation` (helpers `find_joints` / `find_bodies` in `envs/utils/api.py`), **not** `asset.find_joints` / `asset.find_bodies`. Critical for action and observation terms so trained policies transfer across backends.
7. **Contact sensor indices** — mjlab’s contact sensor has no `find_bodies`. Use `find_sensor_bodies(asset, contact_sensor, pattern)` so articulation and sensor indices stay aligned in simulation order.
8. **Contact sensor data fields differ by backend** — do **not** assume a shared `sensor.data.*` API. Isaac uses `ContactSensorData` (`net_forces_w`, `force_matrix_w`, …); mjlab uses `ContactData` (`force`, `found`, …) and only populates fields listed in `ContactSensorCfg.fields`. Branch on `env.backend` or use a thin helper when reading forces / air time.
9. **Scene owns sensors; assets suggest defaults** — prefer task YAML `sensors:` + `envs/sensors/` factories. `AssetSpec.sensors` still seeds robot defaults (e.g. `contact_forces`); task entries with the same name **replace** them. Do not assume every contact sensor is on the robot.
10. **Never smoke-test with the shared root venv** — it is by design incomplete. Use `uv run --project venv/isaac51` (or `isaac60` / `mjlab`). See [.agents/skills/README.md](../README.md#smoke-tests--running-code).
11. **Write ``env.episode_origin`` in ``sample_init``** — after choosing origins (e.g. via `scene.sample_spawn_origin_candidates`), set `env.episode_origin[env_ids] = origins`. Use `episode_origin` (not `scene.env_origins`) for shared 3DGS / episode-local frames. See [Episode origins](#episode-origins).

---

## Checklist: new term

```
Task Progress:
- [ ] Choose term type and base class (Observation / Reward / …)
- [ ] Implement class in the matching envs/mdp/<family>/ module
- [ ] __init__: config kwargs only (no env / asset access)
- [ ] _initialize: super()._initialize(env); then asset/sensor/buffer setup
- [ ] Joint/body indices: `find_joints` / `find_bodies` (simulation order), not `asset.find_*`
- [ ] Contact indices: `find_sensor_bodies` (not `contact_sensor.find_bodies`)
- [ ] Bind sensors by name: `env.scene.sensors[sensor_name]`, entities via `env.scene.entities[entity_name]`
- [ ] Implement the type-specific compute / apply / `_update` API
- [ ] Optional lifecycle: `_update` (Command/Reward; sealed `update`), reset(env_ids, tensordict), pre_step, post_step, startup, debug_draw, **prescribe(tensordict)** / **step()** (command), `in_keys`/`out_keys` when needed
- [ ] If Command/Reward `_update`: **no default args**; missing `in_keys` arrive as `None`
- [ ] If `debug_draw`: use `env.scene.draw_*` / `create_camera_frustum` (not `env.debug_draw`)
- [ ] Optional: symmetry_transform (obs/action) for symaug
- [ ] If overriding `sample_init`: write `env.episode_origin[env_ids] = origins` used for spawn
- [ ] Ensure module is imported (auto-import or explicit in package __init__)
- [ ] Wire into cfg/task/ YAML (and `sensors:` if a new scene sensor is required)
- [ ] Smoke via backend venv (`uv run --project venv/isaac51|mjlab`): instantiate env and step once
```

---

## Scene sensors

Sensors live on the **scene** (`env.scene.sensors`), not on assets. MDP terms only **read** them in `_initialize`.

### Declaration

| Source | Role |
|--------|------|
| `AssetSpec.sensors` (robot factory) | Defaults (Isaac: `dict[str, cfg]`; mjlab: `tuple` of named cfgs) |
| Task YAML `sensors:` | Scene-level sensors; same name **overrides** the asset default |

```yaml
sensors:
  # Preferred: sensor on the RigidObject (single body) + one partner each
  object_robot:
    _target_: contact_sensor
    entity: object
    secondary: robot
  object_ground:
    _target_: contact_sensor
    entity: object
    secondary: terrain

  # Isaac only: several partners on one sensor → force_matrix_w[..., m, :]
  # object_contacts:
  #   _target_: contact_sensor
  #   entity: object
  #   secondary: [terrain, robot]
```

Factories: `active_adaptation/envs/sensors/` (imported from `active_adaptation/__init__.py` and again in backend `setup_scene`). Register with `registry.register("sensor", name, fn)`.

Backends (`isaac/env.py`, `mjlab/env.py`):

1. Attach robot + `AssetSpec.sensors`
2. Spawn `objects:`
3. Build task `sensors:` via `registry.get("sensor", _target_)(backend=..., name=sensor_name, **kwargs)` and merge onto the scene (Isaac after objects so prims exist)

Cameras may still be spawned from obs `edit_spec` (`camera_isaac` / `camera_mjlab`); prefer moving new cameras into `sensors:` factories over time.

### `contact_sensor` factory (`envs/sensors/contact.py`)

Shared kwargs → backend `ContactSensorCfg`:

| Kwarg | Default | Meaning |
|-------|---------|---------|
| `entity` | `robot` | Scene entity to measure |
| `pattern` | `None` | Bodies; see Isaac prim rule below |
| `secondary` | `None` | Partner(s): str or list (`terrain`, `robot`, object keys) |
| `secondary_pattern` | `None` | str or list aligned with `secondary` |
| `track_air_time` | `True` | Air / contact timers |
| `history_length` | `3` | History buffer |
| `fields` / `reduce` | mjlab only | `("found","force")` / `"netforce"` |

**Isaac prim paths**

- Articulation (`robot`): ContactReportAPI is on **child** links → default `{ENV}/Robot/.*`
- RigidObject (`object`, …): ContactReportAPI is on the **root** → default `{ENV}/object` (not `{ENV}/object/.*`)
- Isaac needs `activate_contact_sensors=True` on the asset spawn
- `secondary: terrain` → `/World/ground/terrain/GroundPlane/CollisionPlane` by default (plane). For generator terrain set `secondary_pattern: /World/ground/terrain/mesh`
- Prefer task YAML as `robot_ground` / `robot_object` (`entity: robot`, `secondary: terrain|object`). On **Isaac**, multi-link `Robot/.*` + one partner is auto-inverted to one-to-many (partner primary, robot links as filters) → `force_matrix_w` `(N, 1, B_robot, 3)`. Do **not** use `secondary: [terrain, object]` on one Isaac sensor — use two sensors. **mjlab** keeps `entity: robot` as written.

**mjlab**

- One `ContactMatch` secondary per sensor. Multiple partners → separate `sensors:` entries (factory raises if `secondary` is a list).
- `secondary: terrain` → literal body `terrain` (`entity=None`)

### Binding in MDP terms

```python
def __init__(self, body_names: str, entity_name: str = "robot", sensor_name: str = "contact_forces"):
    ...

def _initialize(self, env):
    super()._initialize(env)
    self.asset = env.scene.entities[self.entity_name]
    self.contact_sensor = env.scene.sensors[self.sensor_name]
    self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
    self.contact_ids = find_sensor_bodies(
        self.asset, self.contact_sensor, self.body_names_pattern
    )[0]
```

Do **not** hardcode `"robot"` / `"contact_forces"` when the task may use object sensors.

**Isaac filtered contacts:** with a non-empty `filter_prim_paths_expr`, pair-specific forces are in `force_matrix_w` `(N, B, M, 3)`; `net_forces_w` remains the unfiltered total. Prefer `force_matrix_w` (sum / any over `M` if needed) for object–ground or object–robot terms. See `observations/contact.py` → `contact_forces` and `uw_manip` `ManipTracking` (`object_robot`).

---

## Cross-backend name / index resolution

Isaac and MuJoCo/mjlab often disagree on joint and body ordering (and contact sensors may list bodies in yet another order). Policies must see a **stable layout** across backends, so asset configs declare authoritative lists:

| Config field | Purpose |
|--------------|---------|
| `asset.cfg.joint_names_simulation` | Canonical joint order for MDP tensors |
| `asset.cfg.body_names_simulation` | Canonical body order for MDP tensors |

**Helpers** (`active_adaptation.envs.utils` / `envs/utils/api.py`):

| Helper | Use for | Notes |
|--------|---------|-------|
| `find_joints(asset, pattern)` | Articulation joint indices | Matches against `joint_names_simulation`, then maps to `asset.joint_names` indices |
| `find_bodies(asset, pattern)` | Articulation body indices | Same for `body_names_simulation` |
| `find_sensor_bodies(asset, contact_sensor, pattern)` | Contact-sensor body indices | Resolves names via `find_bodies` first; Isaac uses `contact_sensor.find_bodies(..., preserve_order=True)`, mjlab indexes `contact_sensor.primary_names` |

**Especially for action and observation terms:** slicing joint/body features in backend-native order silently breaks sim-to-sim / Isaac↔mjlab transfer. Prefer the helpers (or resolve against `*_names_simulation` then `asset.*.index(name)` as in `actions/joint.py`).

Example (reward with feet + contact — see `rewards/gait.py`):

```python
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies

def _initialize(self, env):
    super()._initialize(env)
    self.asset = self.env.scene.articulations["robot"]
    self.contact_sensor = self.env.scene.sensors["contact_forces"]
    self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
    self.body_contact_ids = find_sensor_bodies(
        self.asset, self.contact_sensor, self.body_names_pattern
    )[0]
```

---

## Term APIs (what to implement)

| Type | Config block | Must implement | Notes |
|------|--------------|----------------|-------|
| Observation | `observation.<group>.<name>` | `compute() -> Tensor` | Groups concat along last dim in YAML order; optional `update()` |
| Reward | `reward.<group>.<name>` | `_compute() -> Tensor` or `(rew, is_active)` | Sealed `update` → `_update`; optional `in_keys`/`out_keys`. `compute()` applies `weight` + modifier + EMA |
| Termination | `termination.<name>` | `compute(terminated) -> bool Tensor` or `(term, discount)` | `is_timeout=True` → truncated |
| Action | `input.<name>` | `process_action`, `apply_action` | Set `action_dim`; often `symmetry_transform` |
| Command | `command` | `_update`; optional `prescribe`, `step` | Sealed `update` → `_update`; also `sample_init` (must set `env.episode_origin`); see [Command timing](#command-timing) and [`prescribe`](#prescribe-command-driven-inputs) |
| Randomization | `randomization.<name>` | lifecycle hooks as needed | mjlab: declare `mj_fields` if expanding model |

### Minimal examples

**Reward** (`envs/mdp/rewards/…`):

```python
class my_term(Reward):
    def __init__(self, weight: float, scale: float = 1.0, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.scale = scale

    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    def _compute(self):
        return -self.asset.data.root_com_lin_vel_b[:, 2:3].square() * self.scale
```

**Reward with dynamic gating via `in_keys`** (e.g. `linvel_exp`):

```python
class linvel_exp(Reward):
    in_keys = ["linvel_exp_weight"]  # written by command.update into step TD
    out_keys = None

    def _update(self, weight: torch.Tensor | None) -> None:  # no defaults!
        # cache buffers; missing key → weight is None
        ...

    def _compute(self):
        rew = ...
        return rew, self._weight > 0.0  # is_active from last _update
```

**Observation**:

```python
class my_obs(Observation):
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    def compute(self):
        return self.asset.data.root_link_pose_w.reshape(self.num_envs, -1)
```

**YAML** (class name = registry key; `_target_` optional when key ≠ class name):

```yaml
reward:
  loco:
    my_term: {weight: 1.0, scale: 2.0}

observation:
  policy:
    my_obs: {}
    # or rename: custom_name: {_target_: my_obs, ...}
```

`parse_component_spec`: `_target_` defaults to the YAML key name; remaining keys are kwargs.

---

## Lifecycle and step order

Env construction: `_create_mdp_terms()` (instantiate from cfg) → scene setup → `_initialize_mdp_terms()` → `_build_tensor_specs()`.

Per `_step` (after physics substeps):

1. `command_manager.update(tensordict)` — sealed dispatcher → `_update(*in_keys)`; may write `out_keys`; refresh command state for this step’s rewards
2. `RewardGroup.update(tensordict)` for each group → each reward’s sealed `update` → `_update`
3. `_compute_reward`
4. termination `update` hooks → `_compute_termination`
5. `command_manager.step()` — advance / resample for the **next** step
6. `ObsGroup.update(tensordict)` → `_compute_observation`
7. randomization `update` hooks

**Before physics** (once per env step): `command_manager.prescribe(tensordict)` then each `input_manager.process_action`.

First env init: explicit `startup` once (command → adaptations → obs/reward groups → randomizations → terminations → input managers). Each `_reset`: `_reset_idx` → `scene.reset` → explicit `reset` in the same family order.

### Sealed `update` / `_update` (Command & Reward)

| | Role |
|--|------|
| `update(tensordict)` | **Sealed** (`@final`). Pulls `in_keys` from the step TD (missing → `None`), calls `_update`, writes `out_keys`. Do not override. |
| `_update(*tensors_in)` | Override to refresh caches. Params must match `in_keys` order and **must not have defaults**. |
| `in_keys` / `out_keys` | Optional class attrs (`None` = no TD I/O). |

Enforced at class definition by `check_update_signature` in `envs/mdp/base.py`.

### Command timing

`_update` and `step` are intentionally asymmetric:

| Hook | Role |
|------|------|
| `_update` | Refresh command-dependent tensors from the **current** sim state (and optional `in_keys`) for rewards/terminations this step. |
| `step` | May resample / advance the command for the **upcoming** observation / next reward step. |

**First observation is discarded:** Training treats the first transition after reset as invalid via `is_init` (PPO/off-policy mask with `~is_init`). Do not recompute next-step targets in `reset` solely to validate the post-reset obs.

Pattern:

```python
class MyCommand(Command):
    def _update(self) -> None:
        # errors / caches from active command vs current state
        ...

    def step(self) -> None:
        # advance clip / write targets for upcoming obs
        ...
```

---

## `prescribe`: command-driven inputs

Use **`prescribe(tensordict)`** on `Command` when the command should drive one or more **`task.input`** slots that the policy does **not** learn (reference trajectories, hand-crafted IK targets, scripted gripper phase, etc.).

### When to use

| Pattern | Policy `action` | Command `prescribe` | Example |
|---------|-----------------|---------------------|---------|
| Full policy control | all `input.*` keys | no-op (default) | locomotion |
| Split control | base / throttle only | fills `arm_control`, … | `ManipTracking`: policy → `action`, command → `EndEffectorPose` via `arm_control` |
| Teleop override | may supply keys | only fill **missing** keys | policy can still pass `arm_control` when present |

### Step order (critical)

`prescribe` runs **once at the start of `_step`**, **before** any `input_manager.process_action`:

```python
# env_base._step
self.command_manager.prescribe(tensordict)
for input_key, input_manager in self.input_managers.items():
    input_manager.process_action(tensordict.get(input_key, None))
```

So prescribed values must come from command state already updated on the **previous** step’s `step()` / prior `_update` (same timing as reward targets). Do **not** write next-step references in `prescribe`; use `step()` for that.

| Hook | Writes next-step refs? | Runs `process_action`? |
|------|------------------------|-------------------------|
| `_update` | refreshes current-step caches | no |
| `step` | yes (obs / next reward) | no |
| `prescribe` | no — uses refs from last `step` | fills tensordict only |

### Contract

1. **Key names must match `task.input`** — e.g. `arm_control` for an `EndEffectorPose` action term, not a concatenated policy vector.
2. **Only fill missing keys** — check `tensordict.get(key) is None` before `tensordict.set(...)`. If the learner or teleop already set the key, leave it alone.
3. **Shape must match the action term** — `(num_envs, action_dim)` for that input manager (e.g. 7D pose: pos₃ + quat₄ in body frame).
4. **Default is no-op** — override only when the task needs command-driven actuation.
5. **Not the policy action API** — do not return a single flat action; do not confuse with legacy teleop `get_action`. `prescribe` mutates the step tensordict in place.

### YAML wiring (split control)

```yaml
input:
  arm_control:
    _target_: EndEffectorPose
    joint_names: "arm_joint[1-6]"
    body_name: grasp_point
  action:
    _target_: ConcatenatedAction
    actions:
      - _target_: UnderwaterThrottle
      - _target_: CorrelatedJointPosition
        joint_names: "arm_joint[7,8]"
```

Policy / algo specs use **`action_manager`** (typically `input.action` only). Prescribed keys like `arm_control` are **omitted** from the learned action space.

### Implementation sketch

```python
class MyTracking(Command):
    @override
    def _update(self) -> None:
        # optional: error stats from active refs vs current state
        ...

    @override
    def step(self) -> None:
        # advance clip index; write cmd_eef_pos_b / cmd_eef_quat_b for next step
        self._write_tracking_refs(None, self._raw_time_steps)
        self._refresh_body_frame_commands()

    @override
    def prescribe(self, tensordict: TensorDictBase) -> None:
        if tensordict.get("arm_control") is None:
            tensordict.set(
                "arm_control",
                torch.cat([self.cmd_eef_pos_b, self.cmd_eef_quat_b], dim=-1),
            )
```

Reference: `Command.prescribe` docstring in `envs/mdp/commands/base.py`; example `uw_manip.commands.manip_tracking.ManipTracking`.

### Anti-patterns (`prescribe`)

- Writing next-step trajectory targets in `prescribe` instead of `step`
- Always overwriting input keys (breaks teleop / ablations that pass `arm_control` from outside)
- Putting prescribe logic in `_update` or `reset` (wrong phase; `process_action` never sees it)
- Using a single flat `action` tensor in the command when YAML defines separate `input.*` managers
- Expecting `prescribe` to run inside physics substeps (it runs once per env step, before substeps)

---

## Episode origins

Multi-env layouts and shared appearance (one 3DGS / local frame for all envs) need a clear split:

| Buffer | Owner | Meaning |
|--------|--------|---------|
| `scene.env_origins` | Scene / terrain | Persistent **layout / curriculum** slots (grid or terrain levels) |
| `env.episode_origin` | Env (`(N, 3)`) | Origin used for the **current episode** (what was added in `sample_init`) |

**Contract:** every `sample_init` (including overrides) must write the origins it actually uses:

```python
def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
    origins = self.env.scene.sample_spawn_origin_candidates(env_ids)
    self.env.episode_origin[env_ids] = origins
    init_root_state = self.init_root_state[env_ids]
    init_root_state[:, :3] += origins
    ...
    return init_root_state
```

- `scene.sample_spawn_origin_candidates(env_ids)` — unstamped candidates. Default: `env_origins[env_ids]`. Isaac may random-sample a terrain patch when procedural terrain is active.
- For episode-local math (shared 3DGS, motion refs in env frame, etc.) subtract / add **`env.episode_origin`**, not `scene.env_origins`.
- Base `Command.sample_init` already follows this pattern; overrides that skip `super()` must set `episode_origin` themselves.

`gs_camera` with `origin: env` renders at `mount_pos_w - env.episode_origin` so all envs share one PLY around the episode frame.

---

## Reset API

**Required signature** (all MDP terms):

```python
def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
    ...
```

- Both arguments are required. Most terms leave `tensordict` unused.
- Terms **may read and write** `tensordict` (controlled / curriculum resets, inter-term handoff during the same reset).
- Env always passes a real `TensorDictBase` (allocates an empty one on full reset when the caller passed `None`).
- Order today: `_reset_idx` (`command_manager.sample_init`) → `scene.reset` → explicit `reset` on command, adaptations, obs/reward groups, randomizations, terminations, input managers.
- **Future:** `sample_init` will be removed; `reset` will own initial-state decisions. Do not reintroduce `reset(env_ids)`-only overrides.

---

## Symmetry

For PPO symaug, override `symmetry_transform()` on observation and action terms to return a `SymmetryTransform` matching that term’s vector slice. Groups/managers concat local transforms in config order. If symmetry is undefined, override and raise `NotImplementedError` explicitly.

---

## Backends

Set `supported_backends = ("isaac", "mjlab", …)` on the class when the term is backend-specific. `RegistryMixin.make` returns `None` (and warns) if the active backend is unsupported — the env skips that term.

Cross-backend terms that index joints, bodies, or contacts must still use simulation-order helpers (above); `supported_backends` alone does not fix ordering.

---

## Debug visualization

Use **`env.scene`** (`SceneAdapter`) for all MDP debug drawing. Do **not** use `env.debug_draw` (removed / legacy).

### API

| Call | Purpose |
|------|---------|
| `scene.draw_vector(x, v, size=…, color=…)` | Line segments from `x` along `v` |
| `scene.draw_point(x, color=…, size=…)` | Point cloud |
| `scene.draw_plot(x, size=…, color=…)` | Polyline through points `x` |
| `scene.clear_debug()` | Clear per-step primitives (env inserts this as the first `debug_draw` callback) |
| `scene.create_camera_frustum(name, fov_y=…, aspect=…, scale=…)` | Returns `CameraFrustumHandle` |

`CameraFrustumHandle` accepts torch/numpy for `position`, `wxyz` (WXYZ), and `image` (HWC uint8 RGB/RGBA).

### When callbacks run

`env_base` runs `debug_draw` callbacks when `sim.has_gui()` is true:

| Backend | `has_gui()` true when |
|---------|------------------------|
| Isaac | Omniverse Kit GUI **or** `viewer.viser: true` (`IsaacViserViewer`) |
| mjlab | Not headless (`MjLabViewer` present) |

**GUI vs sensor rendering** (separate `SimAdapter` APIs):

| Call | When | Isaac | mjlab |
|------|------|-------|-------|
| `sim.step()` | every physics substep | physics only | physics only |
| `sim.render_sensors()` | last substep if `env.sensor_render_enabled` | Kit `render()` (cameras) | `Simulation.sense()` |
| `sim.render_gui()` | last substep if GUI, ~30 Hz | Kit viewport (coalesced) + Viser | `MjLabViewer.update()` |

Native camera obs terms set `env.sensor_render_enabled = True` in `_initialize`.

**3DGS / visual world (option A):** appearance is **not** on `sim`. Load via `task.visual` → `env.visual` (`VisualWorld`, e.g. `FvdbGaussianWorld`). Observation `gs_camera` calls `env.visual.render` in `update`/`compute` — no `sensor_render_enabled`. With `origin: env` (default for shared scenes), camera poses are expressed relative to `env.episode_origin` (must be set in `sample_init`; see [Episode origins](#episode-origins)).

**Robot + GS composite:** `_setup_visual` calls `visual.attach_scene_meshes(scene)` for `task.visual.mesh_entities` (default `[robot]`). Isaac pulls body-local visuals via `scene.get_visual_meshes`; mesh RGB-D is rendered by **`simple_raycaster`** (`mesh_renderer: diffrast|raycast`, optional quadric `face_keep`). AA only depth-composites over GS (`envs/visual/mesh_composite.py`). Pass `origin_w=episode_origin` for episode-local cameras.

InteriorGS dirs also ship `{id}_collision.usd` (same frame as the PLY); loaded for Isaac Viser (`/visual/collision`, visible; 3DGS stays hidden). Physics use of that mesh (replace ground plane) is not wired yet — still on `scene`. Placeholder PLY: `envs/visual/fvdb_gs.py` → `PLY_PATH_PLACEHOLDER`.

Gate optional expensive viz with `if self.env.sim.has_gui():` (or a term-local `debug_vis` flag). Prefer `scene.draw_*` unconditionally inside `debug_draw` when the callback is only registered/useful with a GUI — adapters no-op when the viewer is absent.

### Patterns

**Vectors / points** (commands, DVL beams, contact forces):

```python
def debug_draw(self):
    if not self.env.sim.has_gui():
        return
    self.env.scene.draw_vector(starts, vecs, color=(0.1, 0.85, 0.95, 1.0), size=2.0)
    self.env.scene.draw_point(pts, color=(1.0, 0.0, 0.0, 1.0), size=10.0)
```

**Camera image frustum** (e.g. `uw_camera`): register in `_initialize`, push pose+image in `debug_draw`:

```python
# _initialize
self.camera_handle = None
if self.debug_vis:
    self.camera_handle = self.env.scene.create_camera_frustum(
        self.sensor_name, fov_y=fov_y, aspect=aspect
    )

# debug_draw
if self.camera_handle is None:
    return
self.camera_handle.position = pos_w
self.camera_handle.wxyz = quat_wxyz
self.camera_handle.image = image_hwc_uint8
```

Requires a Viser viewer: Isaac `viewer.viser: true`, or mjlab non-headless. Pose from body × mount offset (do not rely on unreliable sensor `pos_w` alone).

### Backend wiring (do not reimplement in terms)

| Piece | Path |
|-------|------|
| Protocol | `envs/adapters.py` |
| Isaac Omni + Viser | `envs/backends/isaac/{adapter,viewer,env}.py` |
| mjlab Viser | `envs/backends/mjlab/{adapter,viewer,env}.py` |

Isaac may draw to Omni DebugDraw and/or Viser in the same call. Mesh upload for Isaac Viser reuses `simple_raycaster.utils_usd` (see raycasting section). Details: [reference.md](reference.md#debug-visualization).

### Anti-patterns (viz)

- Calling `env.debug_draw.*` (use `env.scene.draw_*`)
- Backend-branching (`if backend == "isaac"`) solely for vectors/points/frustums
- Creating frustums every step (register once in `_initialize`)
- Opening Omni Kit UI when only Viser is intended (`has_gui` includes Viser; native Kit checks use the underlying sim)

---

## Raycasting and USD mesh extraction

For Isaac range / depth / occupancy sensing (and for extracting robot visual meshes for Viser), use the installed package **`simple-raycaster`** ([btx0424/simple-raycaster](https://github.com/btx0424/simple-raycaster)) — not Isaac Lab’s single-mesh `RayCaster`. Assume it is available in the uv environment after installing active-adaptation; import as `from simple_raycaster import …`. Package docs: upstream `readme.md` / `AGENTS.md`.

### Which API

| Class | When |
|-------|------|
| `MeshRegistry` + `MultiMeshRaycaster` / `RaycastCamera` | AA scenes: register targets once, refresh poses each step, pass explicit `mesh_pos_w` / `mesh_quat_w` |
| `MultiMeshRaycaster` alone | Standalone scripts, offline USD/MJCF, manual poses |
| `LambertRaycastCameraSensor` + `raycast_camera` obs | MDP RGB-D cameras (shared renderer, per-obs mounts) |

`MeshProximitySensor` / `closest_points` obs: **update required** — pending rewrite on `MeshRegistry` (see `simple_raycaster.MESH_PROXIMITY_UPDATE_REQUIRED`).

Prefer **`raycast_fused`** / **`RaycastCamera.render`** in training loops. Quaternions are **WXYZ**. Call `wp.init()` once per process before the first Warp launch. Use `device="cuda"`.

### Mesh registration (AA)

```python
from active_adaptation.envs.mesh_registry import MeshRegistry

registry = MeshRegistry.for_scene(env.scene, backend=env.backend, device=env.device)
indices = registry.ensure_targets(env.scene, ("terrain", "robot"))
registry.update_poses(env.num_envs)
mesh_pos_w, mesh_quat_w = registry.poses_for_indices(indices)
# MultiMeshRaycaster.raycast_fused(..., mesh_pos_w=..., mesh_quat_w=...)
```

- Entity meshes come from `scene.get_visual_meshes(name)` (body-local); poses from `entity.data.body_link_pose_w`.
- Target filters: `robot/(gripper_.*|base_link)` keeps bodies whose names `re.fullmatch` the regex.
- Isaac `scene.extras[name]` (collision-only `AssetBaseCfg`): mesh + world poses snapped once at registration (static). **Known issue (Isaac Lab 2.3.2):** `XformPrimView.get_world_poses()` may return only env_0 — `MeshRegistry._register_extra` expands via `env_origins` until a proper multi-env query exists.
- Terrain: Isaac uses `scene.ground_mesh`; mjlab uses collision trimesh → `trimesh2wp`.
- Batch `N` on rays must equal `entity.num_instances` (`num_envs`).

### USD → trimesh (body-local visuals / collisions)

Prefer scene adapter APIs (shared helper `envs/backends/isaac/meshes.py`):

```python
visuals = env.scene.get_visual_meshes("robot")       # (body_ids, names, meshes) — skips empty
collisions = env.scene.get_collision_meshes("robot") # same sparse contract; no empty placeholders
```

Low-level extraction still lives in `simple_raycaster.utils_usd` (also used by Viser):

1. Concrete `env_0` paths → `stage.GetPrimAtPath` (AA `envs/backends/isaaclab/meshes.py`). Regex `find_matching_prims` only when the path is a pattern.
2. `get_trimesh_from_prim(prim)` — collect `Mesh`/`Cube` under the prim (follows instance prototypes), convert via `usd2trimesh`, apply **local** transform relative to the parent prim (`world * parent⁻¹`), concatenate + `merge_vertices`.
3. Result is in **body / parent frame**; at runtime multiply by `body_link_pose_w` (or use `MeshRegistry.update_poses`).
4. Visuals: `{body}/visuals` then body root. Collisions: `{body}/collisions` then `{body}/collision`.

Static terrain: combine under e.g. `/World/ground` and keep identity pose (geometry already world-framed).

mjlab: `env.scene.get_visual_meshes` / `get_collision_meshes` → `envs/backends/mjlab/meshes.py` (contype/conaffinity role split + `mjviser.conversions.merge_geoms`). Low-level: `simple_raycaster.utils_mjc.get_trimesh_from_body` (mesh geoms only, no role filter).

### In-repo usage patterns

| Pattern | Where | Notes |
|---------|--------|------|
| `LambertRaycastCameraSensor` + `raycast_camera` obs | `sensors/camera.py`, `observations/extero/camera.py` | Shared renderer; obs owns mount |
| `MeshRegistry` + explicit poses | `mesh_registry.py` | Geometry + pose cache for Warp consumers |
| V1 + `env.ground_mesh` | `height_scan`, DVL in `underwater.py` | Ground-only or manual poses; DVL still uses IsaacLab `raycast_mesh` on `ground_mesh` |
| Scene ground Warp mesh | `IsaacSceneAdapter.ground_mesh` | Plane or USD mesh at `/World/ground` |

Isaac/mjlab **Viser** robot meshes (viewer internals, not MDP terms): same body-visual extraction as `add_isaac_entity` / `get_trimesh_from_prim`; upload once; each `viewer.update` writes `body_link_pose_w`. Camera obs debug uses `scene.create_camera_frustum`, not flat GUI image panels alone. See [reference.md](reference.md#debug-visualization).

### Anti-patterns (raycast / mesh)

- Using Isaac Lab `RayCaster` when multiple dynamic meshes are needed
- Passing world-space meshes into entity slots (meshes must be body-local)
- Extracting USD before `AppLauncher` / stage exists (`from pxr import Usd` needs Isaac or standalone `usd-core`)
- CPU Warp device for batched training raycasts
- Forgetting `wp.init()` or re-adding meshes without re-`initialize()` (mesh-id array stale)

---

## Anti-patterns

- Accessing `env.scene` / assets in `__init__` (scene does not exist yet)
- Overriding sealed `Command.update` / `Reward.update` (implement `_update`; no default args)
- Putting command resampling / next-step target writes in `_update` (use `step`)
- Writing reference inputs in `_update` / `reset` instead of `prescribe` + `step` (see [`prescribe`](#prescribe-command-driven-inputs))
- Overwriting prescribed `task.input` keys in `prescribe` when the key is already set (breaks policy override / teleop)
- Re-evaluating next-step command/obs targets in `reset` “to fix” the first obs (that obs is discarded via `is_init`)
- Overriding `sample_init` without writing `env.episode_origin[env_ids]` (shared 3DGS / episode-local consumers break)
- Subtracting `scene.env_origins` for episode-local frames when spawn was randomized — use `env.episode_origin`
- Forgetting to import the module (class never enters the registry)
- Returning unbatched reward/obs tensors
- Using `env.debug_draw` instead of `env.scene.draw_*` / `create_camera_frustum`
- Using `asset.find_joints` / `asset.find_bodies` for MDP feature layout (backend-native order; breaks Isaac↔mjlab transfer) — use `find_joints` / `find_bodies` / `*_names_simulation`
- Calling `contact_sensor.find_bodies` directly (missing on mjlab; may disagree with articulation order even on Isaac) — use `find_sensor_bodies`
- Assuming Isaac/mjlab contact `sensor.data` share field names (`net_forces_w` vs `force`) — see [reference.md](reference.md#contact-sensor-data-fields-isaac--mjlab)
- Hardcoding `scene.sensors["contact_forces"]` / `articulations["robot"]` when the task may use object or filtered sensors — take `sensor_name` / `entity_name`
- Isaac contact on a RigidObject with `prim_path={ENV}/object/.*` (children lack ContactReportAPI) — use the entity root; enable `activate_contact_sensors` on spawn
- Filtering Isaac terrain with `/World/ground/terrain` alone (Xform) — use GroundPlane/CollisionPlane or mesh leaves
- Reading Isaac `net_forces_w` when you need filtered object–ground forces — use `force_matrix_w`
