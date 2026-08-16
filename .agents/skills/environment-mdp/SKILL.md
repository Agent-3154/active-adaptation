---
name: environment-mdp
description: Implement and wire MDP terms in active-adaptation (observations, rewards, terminations, actions, commands, randomizations) using the V2 deferred-init API. Use when adding or modifying classes under envs/mdp/, editing cfg/task/ observation/reward/termination/input/command/randomization/sensors blocks, registering scene sensors (envs/sensors/, contact_sensor), debugging env_base step/reset callbacks, cross-backend contact sensor data (Isaac net_forces_w / force_matrix_w vs mjlab force), debug viz (scene.draw_*, camera frustums), Warp/simple-raycaster raycasting, USD mesh extraction, or Isaac/mjlab Viser viewers.
---

# Environment / MDP terms (active-adaptation)

Implement MDP terms as **V2** classes: construct from Hydra kwargs **without** an env, then bind in `_initialize(env)` after the scene exists. Register via `RegistryMixin`; task YAML selects them by class name.

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
- Porting a legacy `Reward`/`Observation`/… term to V2
- Wiring or renaming terms in `cfg/task/`
- Declaring task-level scene sensors (`sensors:` — object contact, filtered ground contact, …)
- Debugging missing callbacks, wrong step order, or registry lookup failures
- Debug vectors / points / plots / camera frustums across Isaac and mjlab
- Isaac multi-mesh raycasting, USD → trimesh/Warp extraction, or browser Viser robot meshes

---

## Hard rules

1. **Use V2 only** — `ObservationV2`, `RewardV2`, `TerminationV2`, `ActionV2`, `CommandV2`, `RandomizationV2`. Legacy bases that take `env` in `__init__` are deprecated (`Reward` raises).
2. **Two-phase init** — `__init__(**cfg_kwargs)` stores config only; `_initialize(env)` binds `self.env`, caches assets/sensors, allocates buffers. Always call `super()._initialize(env)` first when overriding.
3. **Register by subclassing** — subclassing a V2 base auto-registers under the class name (or `namespace.ClassName`). Import the module so registration runs (see package `__init__.py` auto-import patterns).
4. **Shape convention** — tensors are batched `(num_envs, …)`. Rewards/terminations usually return `(num_envs, 1)`.
5. **Override only what you need** — `_add_mdp_component` registers `startup` / `reset` / `update` / `pre_step` / `post_step` / `debug_draw` only when the subclass **overrides** the base method (`is_method_implemented`). Empty overrides still register.
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
- [ ] Choose term type and V2 base class
- [ ] Implement class in the matching envs/mdp/<family>/ module
- [ ] __init__: config kwargs only (no env / asset access)
- [ ] _initialize: super()._initialize(env); then asset/sensor/buffer setup
- [ ] Joint/body indices: `find_joints` / `find_bodies` (simulation order), not `asset.find_*`
- [ ] Contact indices: `find_sensor_bodies` (not `contact_sensor.find_bodies`)
- [ ] Bind sensors by name: `env.scene.sensors[sensor_name]`, entities via `env.scene.entities[entity_name]`
- [ ] Implement the type-specific compute / apply / sync API
- [ ] Optional lifecycle: update, reset(env_ids, tensordict), pre_step, post_step, startup, debug_draw
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
  object_contact:
    _target_: contact_sensor   # registry group "sensor"
    entity: object             # scene entity key (objects.<name> / robot)
    secondary_entity: terrain  # optional filter (ground / another entity)
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
| `secondary_entity` | `None` | Contact filter (`terrain` or another entity key) |
| `secondary_pattern` | `None` | Override filter prim / mjlab match |
| `track_air_time` | `True` | Air / contact timers |
| `history_length` | `3` | History buffer |
| `fields` / `reduce` | mjlab only | `("found","force")` / `"netforce"` |

**Isaac prim paths**

- Articulation (`robot`): ContactReportAPI is on **child** links → default `{ENV}/Robot/.*`
- RigidObject (`object`, …): ContactReportAPI is on the **root** → default `{ENV}/object` (not `{ENV}/object/.*`)
- Isaac needs `activate_contact_sensors=True` on the asset spawn
- `secondary_entity: terrain` → filters `/World/ground/terrain/GroundPlane/CollisionPlane` (plane) and `/World/ground/terrain/mesh` (generator). Parent `/World/ground/terrain` is an Xform, not the collision leaf.

**mjlab**

- `ContactMatch(entity=entity, pattern=...)` for primary
- `secondary_entity: terrain` → literal body `terrain` (`entity=None`)

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

**Isaac filtered contacts:** with a non-empty `filter_prim_paths_expr`, pair-specific forces are in `force_matrix_w` `(N, B, M, 3)`; `net_forces_w` remains the unfiltered total. Prefer `force_matrix_w` (sum over `M` if needed) for object–ground terms. See `observations/contact.py` → `contact_forces`.

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

| Type | Config block | Factory | Must implement | Notes |
|------|--------------|---------|----------------|-------|
| Observation | `observation.<group>.<name>` | `ObservationV2.make` | `compute() -> Tensor` | Groups concat along last dim in YAML order |
| Reward | `reward.<group>.<name>` | `RewardV2.make` | `_compute() -> Tensor` or `(rew, is_active)` | `compute()` applies `weight` + modifier + EMA |
| Termination | `termination.<name>` | `TerminationV2.make` | `compute(terminated) -> bool Tensor` or `(term, discount)` | `is_timeout=True` → truncated |
| Action | `input.<name>` | `ActionV2.make` | `process_action`, `apply_action` | Set `action_dim`; often `symmetry_transform` |
| Command | `command` | `CommandV2.make` | `sync_state`, `update` | Also `sample_init` (must set `env.episode_origin`); see step order |
| Randomization | `randomization.<name>` | `RandomizationV2.make` | lifecycle hooks as needed | mjlab: declare `mj_fields` if expanding model |

### Minimal examples

**Reward** (`envs/mdp/rewards/…`):

```python
class my_term(RewardV2):
    def __init__(self, weight: float, scale: float = 1.0, track_var: bool = False):
        super().__init__(weight, track_var=track_var)
        self.scale = scale

    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    def _compute(self):
        return -self.asset.data.root_com_lin_vel_b[:, 2:3].square() * self.scale
```

**Observation**:

```python
class my_obs(ObservationV2):
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

Env construction: `_create_mdp_terms()` (instantiate from cfg) → `_setup_simulation()` → `_initialize_mdp_terms()` → `_build_tensor_specs()`.

Per `_step` (after physics substeps):

1. `command_manager.sync_state()` — refresh intermediates from the active command; **do not** change next-step targets
2. `update` callbacks (obs/reward/rand/… that override `update`)
3. rewards → terminations (read last step’s next-step command as this step’s target)
4. `command_manager.update()` — may resample; write **next-step** targets
5. observations (see post-`update` command)

Inside each physics substep: `process_action` (once) → `apply_action` → `pre_step` → sim → `post_step`.

First `_reset`: run `startup` callbacks once, then for reset envs: `_reset_idx` → `scene.reset` → `reset` callbacks.

### Command timing

`sync_state` and `update` are intentionally asymmetric:

| Hook | Role |
|------|------|
| `sync_state` | Refresh **intermediates** derived from the **already active** command (errors, caches) for rewards/terminations. **Do not** change the command or write next-step targets. |
| `update` | May resample / advance the command. Write **next-step** targets that observations will see. |

**Next-step targets:** Command buffers used by observations (and by the *following* step's rewards) are written in `update` only. Rewards in the current step read whatever `update` produced on the **previous** step — that previous "next-step" command is now the target for the state just simulated. Do **not** recompute those targets in `sync_state` or `reset` just to make the post-reset observation look valid.

**First observation is discarded:** Training treats the first transition after reset as invalid via `is_init` (PPO/off-policy mask with `~is_init`). That avoids paying for a second command/obs evaluation on reset: leave next-step targets to the first `update` after the episode starts.

Pattern for look-ahead / trajectory commands:

```python
def sync_state(self):
    # optional: error stats from last update's targets vs current state
    ...

def update(self):
    # write targets for the upcoming obs / next reward step
    ...
```

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
- Base `Command` / `CommandV2.sample_init` already follows this pattern; overrides that skip `super()` must set `episode_origin` themselves.

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
- Order today: `_reset_idx` (`command_manager.sample_init`) → `scene.reset` → term `reset` callbacks.
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
| `MultiMeshRaycasterV2` | Inside Isaac Lab: register static + entity meshes once; poses come from `entity.data.body_link_pose_w` |
| `MultiMeshRaycaster` | Manual poses, offline USD/MJCF, or incremental `add_from_path` |

Prefer **`raycast_fused`** in training loops. Quaternions are **WXYZ**. Call `wp.init()` once per process before the first Warp launch. Use `device="cuda"`.

### Isaac registration (V2)

In `_initialize` (scene must exist):

```python
from simple_raycaster import MultiMeshRaycasterV2

self.raycaster = MultiMeshRaycasterV2(device=self.device)
self.raycaster.add_isaac_static("/World/ground")          # world-baked static mesh, identity pose
self.raycaster.add_isaac_entity(self.env.scene.articulations["robot"])  # one mesh per body
# hit_pos, hit_dist = self.raycaster.raycast_fused(ray_starts_w, ray_dirs_w, min_dist=..., max_dist=...)
```

- `add_isaac_entity` loads `{body}/visuals` under `entity.root_physx_view.prim_paths[0]`; matched visual count **must** equal `entity.num_bodies`.
- Batch `N` on rays must equal `entity.num_instances` (`num_envs`).
- Do not call V2’s private `_add_mesh` / `_add_from_path` without updating `entities` (registration validation will fail).

### USD → trimesh (body-local visuals / collisions)

Prefer scene adapter APIs (shared helper `envs/backends/isaac/meshes.py`):

```python
visuals = env.scene.get_visual_meshes("robot")       # list[trimesh], body_names order
collisions = env.scene.get_collision_meshes("robot") # empty mesh if body has no collision prim
```

Low-level extraction still lives in `simple_raycaster.utils_usd` (also used by Viser):

1. `find_matching_prims(regex, stage)` — stage traverse with anchored regex.
2. `get_trimesh_from_prim(prim)` — collect `Mesh`/`Cube` under the prim (follows instance prototypes), convert via `usd2trimesh`, apply **local** transform relative to the parent prim (`world * parent⁻¹`), concatenate + `merge_vertices`.
3. Result is in **body / parent frame**; at runtime multiply by `body_link_pose_w` (or let V2 do it).
4. Visuals: `{body}/visuals` (required). Collisions: `{body}/collisions` then `{body}/collision`.

Static terrain: combine under e.g. `/World/ground` and keep identity pose (geometry already world-framed).

mjlab: `env.scene.get_visual_meshes` / `get_collision_meshes` → `envs/backends/mjlab/meshes.py` (contype/conaffinity role split + `mjviser.conversions.merge_geoms`). Low-level: `simple_raycaster.utils_mjc.get_trimesh_from_body` (mesh geoms only, no role filter).

### In-repo usage patterns

| Pattern | Where | Notes |
|---------|--------|------|
| V2 entity + static | `observations/extero.py` → `raycast_camera` | Preferred for robots/objects |
| V1 + `env.ground_mesh` | `height_scan`, DVL in `underwater.py` | Ground-only or manual poses; DVL still uses IsaacLab `raycast_mesh` on `ground_mesh` |
| Scene ground Warp mesh | `IsaacSceneAdapter.ground_mesh` | Plane or USD mesh at `/World/ground` |

Isaac/mjlab **Viser** robot meshes (viewer internals, not MDP terms): same body-visual extraction as `add_isaac_entity` / `get_trimesh_from_prim`; upload once; each `viewer.update` writes `body_link_pose_w`. Camera obs debug uses `scene.create_camera_frustum`, not flat GUI image panels alone. See [reference.md](reference.md#debug-visualization).

### Anti-patterns (raycast / mesh)

- Using Isaac Lab `RayCaster` when multiple dynamic meshes are needed
- Passing world-space meshes into V2 entity slots (entity meshes must be body-local)
- Extracting USD before `AppLauncher` / stage exists (`from pxr import Usd` needs Isaac or standalone `usd-core`)
- CPU Warp device for batched training raycasts
- Forgetting `wp.init()` or re-adding meshes without re-`initialize()` (mesh-id array stale)

---

## Anti-patterns

- Accessing `env.scene` / assets in `__init__` (scene does not exist yet)
- Subclassing legacy `Reward` / `Observation` / …
- Putting command resampling or next-step target writes in `sync_state` (use `update`)
- Re-evaluating next-step command/obs targets in `reset` “to fix” the first obs (that obs is discarded via `is_init`)
- Overriding `sample_init` without writing `env.episode_origin[env_ids]` (shared 3DGS / episode-local consumers break)
- Subtracting `scene.env_origins` for episode-local frames when spawn was randomized — use `env.episode_origin`
- Forgetting to import the module (class never enters the registry)
- Returning unbatched reward/obs tensors
- Registering empty `update`/`reset` overrides unintentionally (they still run every step/reset)
- Using `env.debug_draw` instead of `env.scene.draw_*` / `create_camera_frustum`
- Using `asset.find_joints` / `asset.find_bodies` for MDP feature layout (backend-native order; breaks Isaac↔mjlab transfer) — use `find_joints` / `find_bodies` / `*_names_simulation`
- Calling `contact_sensor.find_bodies` directly (missing on mjlab; may disagree with articulation order even on Isaac) — use `find_sensor_bodies`
- Assuming Isaac/mjlab contact `sensor.data` share field names (`net_forces_w` vs `force`) — see [reference.md](reference.md#contact-sensor-data-fields-isaac--mjlab)
- Hardcoding `scene.sensors["contact_forces"]` / `articulations["robot"]` when the task may use object or filtered sensors — take `sensor_name` / `entity_name`
- Isaac contact on a RigidObject with `prim_path={ENV}/object/.*` (children lack ContactReportAPI) — use the entity root; enable `activate_contact_sensors` on spawn
- Filtering Isaac terrain with `/World/ground/terrain` alone (Xform) — use GroundPlane/CollisionPlane or mesh leaves
- Reading Isaac `net_forces_w` when you need filtered object–ground forces — use `force_matrix_w`
