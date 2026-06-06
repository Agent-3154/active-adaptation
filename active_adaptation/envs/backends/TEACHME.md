# Multi-backend simulation support in active-adaptation

## Directory structure

```
envs/backends/
├── isaac/
│   ├── env.py          # IsaacBackendEnv
│   └── adapter.py      # IsaacSimAdapter, IsaacSceneAdapter
├── mjlab/
│   ├── env.py          # MjlabBackendEnv
│   ├── adapter.py      # MjlabSimAdapter, MjlabSceneAdapter
│   └── viewer.py       # MjLabViewer (Viser-based)
├── mujoco/
│   ├── env.py          # MujocoBackendEnv
│   ├── adapter.py      # MujocoSimAdapter, MujocoSceneAdapter
│   └── mujoco.py       # custom MJScene, MJArticulation, MJSim
└── motrix/
    ├── env.py          # MotrixBackendEnv (planned)
    ├── adapter.py      # MotrixSimAdapter, MotrixSceneAdapter (stub)
    └── test.py         # local smoke test (stub)
```

Key files *outside* `backends/`:

| Path | Role |
|------|------|
| `envs/env_base.py` | Shared TorchRL env + MDP loop (`_EnvBase`) |
| `envs/adapters.py` | `SimAdapter` / `SceneAdapter` Protocol definitions |
| `envs/utils/api.py` | Backend-agnostic body/joint/sensor lookup helpers |
| `assets/asset_cfg.py` | Backend-specific asset conversion (`isaaclab()`, `mujoco()`, `mjlab()`) |
| `active_adaptation/__init__.py` | `set_backend` / `get_backend`, `init()` |
| `helpers.py` | `make_env_policy()` env factory |
| `cfg/train.yaml` | `backend: isaac` default |

---

## High-level architecture

The design is **adapter + thin backend env subclasses**. Shared logic lives in `_EnvBase`; each backend subclass only overrides `setup_scene()` to build native sim/scene objects and wrap them in adapters.

```
torchrl.envs.EnvBase
    └── RegistryMixin
            └── _EnvBase                    # shared MDP + step loop
                    ├── IsaacBackendEnv     # Isaac Lab / Isaac Sim
                    ├── MujocoBackendEnv    # single-env raw MuJoCo
                    ├── MjlabBackendEnv     # parallel MuJoCo via mjlab
                    └── MotrixBackendEnv    # parallel MuJoCo via motrixsim (planned)
```

`_EnvBase` holds:
- `self.sim: SimAdapter` — physics engine handle
- `self.scene: SceneAdapter` — scene/articulation manager

After `setup_scene()`, `_EnvBase` casts both to the adapter protocols (structural typing via `cast`).

---

## Adapter contracts (`envs/adapters.py`)

Defined as `typing.Protocol` — **no inheritance required**, duck-typed.

### `SimAdapter`

| Method | Description |
|--------|-------------|
| `get_physics_dt()` | physics timestep |
| `has_gui()` | whether a viewer is open |
| `step(render=False)` | advance physics |
| `render()` | update viewport |
| `set_camera_view(eye, target)` | position camera |

Unknown attributes fall through via `__getattr__` to the wrapped native sim object.

### `SceneAdapter`

| Attribute/Method | Description |
|-----------------|-------------|
| `num_envs`, `reset(env_ids)` | basic multi-env interface |
| `update(dt)`, `write_data_to_sim()` | sync state between MDP and sim |
| `articulations`, `sensors`, `env_origins` | entity accessors |
| `zero_external_wrenches()` | clear applied forces |
| `ground_mesh` | Warp mesh for height-field raycasts |
| `get_spawn_origins(env_ids)` | terrain-based spawn (Isaac override) |
| `create_sphere_marker` / `create_arrow_marker` | debug visualization |

---

## Backend selection

### 1. Global flag

`active_adaptation/__init__.py` exposes:

```python
aa.set_backend("isaac" | "mujoco" | "mjlab" | "motrix")   # call once per process
aa.get_backend()                                             # read everywhere
```

`aa.init(cfg)` (called from training scripts):
1. Sets backend from Hydra `cfg.backend` (default `"isaac"`)
2. Forces `cfg.device = "cuda"` for mjlab, `"cpu"` for mujoco *(motrix TBD — likely CPU/numpy batch)*
3. Launches `AppLauncher` (Isaac only) before any Isaac imports

### 2. Env class selection

`make_env_policy()` in `helpers.py` picks the right subclass:

```python
if backend == "isaac":
    env_cls = _EnvBase.registry[cfg.task.get("env_class", "IsaacBackendEnv")]
elif backend == "mujoco":
    env_cls = _EnvBase.registry[cfg.task.get("env_class", "MujocoBackendEnv")]
    cfg.task.num_envs = 1
    cfg.task.reward = {}   # rewards disabled in mujoco (debug/validation mode)
elif backend == "mjlab":
    env_cls = _EnvBase.registry[cfg.task.get("env_class", "MjlabBackendEnv")]
elif backend == "motrix":
    env_cls = _EnvBase.registry[cfg.task.get("env_class", "MotrixBackendEnv")]  # planned
```

`cfg.task.env_class` can override the default class per backend.

### 3. Asset configuration

`AssetCfg` (in `assets/asset_cfg.py`) has per-backend factory methods:

| Backend | Method |
|---------|--------|
| Isaac | `AssetCfg.isaaclab()` |
| MuJoCo | `AssetCfg.mujoco()` → `MJArticulationCfg` |
| mjlab | `AssetCfg.mjlab()` |
| motrix | `AssetCfg.motrix()` *(planned)* |

The file is **import-time backend-specific** — it branches on `aa.get_backend()` to import the right native types.

### 4. MDP component filtering

MDP pieces (`Observation`, `Reward`, etc.) subclass `RegistryMixin` and may declare:

```python
supported_backends = ("isaac", "mjlab")  # only instantiated on these backends
```

`RegistryMixin.make()` skips (warns, returns `None`) components not supported on the current backend.

---

## Concrete backend implementations

### Isaac (`isaac/`)

- **Native stack**: Isaac Lab `SimulationContext`, `InteractiveScene`
- **`setup_scene()`**: builds scene from asset registry + terrain + optional objects; wires dome light + optional Replicator RGB annotator
- **Parallelism**: `cfg.num_envs` via Isaac Lab scene replication
- **Extras**: `VisualizationMarkers`, `debug_draw`, terrain spawn origins, USD ground mesh

### MuJoCo (`mujoco/`)

- **Native stack**: custom `MJSim` / `MJScene` / `MJArticulation` in `mujoco.py` (Isaac-like API)
- **Single env only** (`num_envs = 1`); uses passive `mujoco.viewer`
- **Purpose**: debug/validation; rewards disabled in `make_env_policy`

### mjlab (`mjlab/`)

- **Native stack**: `mjlab.scene.Scene`, `mjlab.sim.Simulation` (GPU-parallel)
- **Viewer**: `MjLabViewer` (Viser-based), driven from env step
- **Ground mesh**: built from terrain geoms as a Warp mesh for raycasts
- **Step quirk**: `sim.render()` is *not* called inside the step loop (mjlab handles rendering separately via viewer)

### motrix (`motrix/`)

- **Status**: stub only (`adapter.py`, `test.py`); backend not wired into `aa.init()` / `make_env_policy()` yet.
- **Closest analogue**: mjlab — GPU-friendly batched MuJoCo-style sim, but via the `motrixsim` Python package instead of mjlab.
- **Package name**: `motrixsim` (import as `import motrixsim as mx`). Do **not** use `import motrix` — that name is wrong in the current stub.

See [MotrixSim backend reference](#motrixsim-backend-reference) below for API details distilled from `motrix/motrixsim-docs/examples/`.

---

## Simulation step loop (`_EnvBase._step`)

Backend-agnostic:

1. `input_managers.process_action()`
2. For each `decimation` substep:
   - `scene.zero_external_wrenches()`
   - apply actions + pre_step callbacks
   - `scene.write_data_to_sim()`
   - `sim.step(render=False)`
   - `scene.update(physics_dt)` + post_step callbacks
3. Render (skipped inside loop for mjlab)
4. Compute rewards, terminations, observations
5. Debug draw (Isaac `debug_draw` or mjlab viewer update)

---

## Adding a new backend (extension guide)

1. Add `"newbackend"` to `set_backend` in `active_adaptation/__init__.py`
2. Create `backends/newbackend/env.py` with `NewBackendEnv(_EnvBase)` implementing `setup_scene()`
3. Create `backends/newbackend/adapter.py` with `NewSimAdapter` and `NewSceneAdapter` satisfying the protocols
4. Add `AssetCfg.newbackend()` converter in `assets/asset_cfg.py`
5. Add a branch in `make_env_policy()` in `helpers.py`
6. Annotate MDP components with `supported_backends` where needed

For motrix specifically, follow the [step-by-step plan](#motrix-backend-implementation-plan) at the end of this doc.

---

## MotrixSim backend reference

Source: `motrix/motrixsim-docs/examples/` and `motrix/motrixsim-docs/docs/source/en/user_guide/main_function/model_building.md`.

### Overview

MotrixSim splits simulation into two layers:

| Layer | Type | Role |
|-------|------|------|
| Model (immutable) | `SceneModel` | Compiled scene graph: bodies, joints, actuators, sensors, geoms, options |
| Data (mutable) | `SceneData` | Per-step state: dof pos/vel, actuator ctrls, external forces |

Physics and rendering are **decoupled**:

```
SceneModel  ──build/load──▶  immutable compiled model
SceneData   ──per instance──▶  mutable state (single or batched)
model.step(data)             advance physics
RenderApp.sync(data)         update viewport (optional, separate loop)
```

Minimal loop (`getting_started/hello_motrixsim.py`, `getting_started/falling_ball.py`):

```python
import motrixsim as mx

model = mx.load_model("path/to/scene.xml")
data = mx.SceneData(model)
with mx.render.RenderApp() as render:
    render.launch(model)
    while True:
        model.step(data)   # or mx.step(model, data)
        render.sync(data)
```

Timed loop helper: `motrixsim.run.render_loop(phys_dt, render_fps, physics_fn, render_fn)`.

### Loading and building models

Two entry points:

| API | Use case | Example |
|-----|----------|---------|
| `load_model(path)` | Load a ready-made MJCF/XML scene file | `hello_motrixsim.py`, `go1.py` |
| `msd.from_file(path).build()` | Programmatic composition (attach robots, terrain, cameras) | `robot_locomotion.py`, `combine_msd.py` |

**MSD workflow** (`motrixsim.msd`):

```python
import motrixsim as mx

scene = mx.msd.from_file("examples/assets/common/flat_scene_with_ssgi.xml")
robot = mx.msd.from_file("examples/assets/go2/go2_mjx.xml")
scene.attach(robot)
model = scene.build()
```

`World.attach()` supports:

- `other_translation`, `other_rotation` — placement
- `other_prefix` / suffixes — avoid name collisions when instancing
- `self_link_name` + `other_link_name` — attach a subtree at a specific link
- `msd.from_str(mjcf_string)` — inline MJCF (e.g. follower camera in `robot_locomotion.py`)

After `build()`, the result is an immutable `SceneModel`. All runtime mutation goes through `SceneData`.

### Parallel / batched simulation

MotrixSim supports vectorized multi-instance physics via a batch dimension on `SceneData`.

```python
batch_size = 1024
model = load_model(path)
data = SceneData(model, batch=(batch_size,))

# All state arrays gain a leading batch dim:
assert data.dof_pos.shape == (batch_size, model.num_dof_pos)

for _ in range(steps):
    model.step(data)          # steps all instances at once
```

Key examples:

| File | What it demonstrates |
|------|---------------------|
| `parallel/parallelsim.py` | Batch ctrls, reset, masked writes (`data[mask]`) |
| `parallel/parallel_bench.py` | Headless throughput benchmark (`--batch 1024`) |
| `physics/model.py` | Per-instance actuator ctrls, link poses |
| `randomize/*.py` | Per-instance physics parameter overrides |

**Important batch semantics** (`parallel/parallelsim.py`):

- `data.reset(model)` — reset all instances
- `data[i]` — single-instance view (shape `()`)
- `data[mask]` — masked batch view for selective writes
- `data.actuator_ctrls = np.random.rand(batch_size, model.num_actuators)` — batch control

**Render vs physics spacing**: `render.launch(model, batch=N, render_offset=[...])` offsets *visual* instances only. Physics instances remain at the origin unless you explicitly set root poses. This differs from mjlab/Isaac `env_spacing` — the motrix backend must handle env origins explicitly (e.g. write root translations into `SceneData` per env, or attach robots at different MSD translations before `build()`).

### Physics options

Accessed via `model.options` (`physics/options.py`):

| Property | Notes |
|----------|-------|
| `timestep` | Physics dt → maps to `SimAdapter.get_physics_dt()` |
| `gravity` | `[x, y, z]` vector, writable at runtime |
| `max_iterations`, `solver_tolerance` | Solver settings |
| `disable_gravity`, `disable_contacts` | Simulation flags |

### Entity API (bodies, links, joints, actuators)

Reference examples under `examples/physics/`:

| Module | Key APIs |
|--------|----------|
| `body.py` | `model.get_body(name)`, `body.get_pose(data)`, `body.floatingbase`, `set_translation` / `set_rotation`, `get_joint_dof_pos/vel`, `set_actuator_ctrls` |
| `link.py` | Link poses, `model.get_link_poses(data)` |
| `joint.py` | `model.get_joint(name)`, `joint.set_dof_pos/vel`, `model.joint_dof_vel_indices` |
| `actuator.py` | `model.get_actuator(name)`, `actuator.set_ctrl(data, val)`, `model.get_actuator_ctrls(data)`, `model.actuator_ctrl_limits` |
| `geom.py` | Geom access, friction overrides |
| `site_and_sensor.py` | Sites, `model.get_sensor_value(name, data)`, contact sensor layout |
| `external_force.py` | `link.add_external_force/torque(data, vec, local=, point=)` |
| `hfield.py` | `model.get_hfield(name)`, `height_matrix`, `GeomHField` |

**Pose convention**: 7D `[x, y, z, qx, qy, qz, qw]`. Gravity in body frame: `body.get_rotation_mat(data).T @ [0, 0, -1]` (see `utils/robot.py`).

**Actuator control pattern** (locomotion robots use MJCF `<position>` actuators):

```python
# Single actuator
actuator = model.get_actuator("FR_hip")
actuator.set_ctrl(data, target_angle)

# All actuators on a body
body.set_actuator_ctrls(data, ctrl_array)   # shape (num_actuators,) or (batch, num_actuators)
```

### Sensors

Named sensors in MJCF, read after stepping:

```python
lin_vel = model.get_sensor_value("local_linvel", data)   # 3D
gyro    = model.get_sensor_value("gyro", data)           # 3D
contact = model.get_sensor_value("box_floor_contact", data)  # [found, 12 * num_contacts]
```

Robot-specific sensor names vary by asset (see `utils/robot.py`):

| Robot | Base link | Linvel sensor | Gyro sensor | MJCF path (examples) |
|-------|-----------|---------------|-------------|----------------------|
| Go1 | `trunk` | `local_linvel` | `gyro` | `assets/go1/go1_mjx_fullcollisions.xml` |
| Go2 | `base` | `local_linvel` | `gyro` | `assets/go2/go2_mjx.xml` |
| G1 | `pelvis` | `local_linvel_pelvis` | `gyro_pelvis` | `assets/g1/g1.xml` |
| G1 12-DoF | `pelvis` | `local_linvel` | `gyro` | `assets/g1/g1_12dof.xml` |
| Spot | (see spot.xml) | — | — | `assets/boston_dynamics_spot/spot.xml` |

### Domain randomization (per-batch overrides)

Examples in `examples/randomize/`:

| Example | API |
|---------|-----|
| `mass.py` | `link.set_mass_override(data, mass_array)` |
| `friction.py` | `geom.set_friction_override(data, friction_array)` |
| `frictionloss.py` | Joint frictionloss override |
| `armature.py` | Joint armature override |
| `actuator_kp_kd.py` | Actuator gain override |
| `geom_size.py` | Geom size override |
| `com.py` | Center-of-mass override |
| `gravity_direction.py` | Gravity direction override |

All overrides accept batch-shaped numpy arrays matching `SceneData` batch dim.

### Terrain and height queries

- Height fields: `model.get_hfield("terrain1")`, `.height_matrix`, `.get(row, col)` (`physics/hfield.py`)
- `GeomHField` geom type for collision meshes derived from hfields
- `TerrainScanner(terrain, frame, offsets, alignment, output="height")` — ray/grid height sampling relative to a link (`control/robot_locomotion.py`)
- Stairs terrain: `examples/assets/common/terrain_stairs.xml`

For `SceneAdapter.ground_mesh` (Warp raycasts), likely build a Warp mesh from `GeomHField.height_matrix` or terrain geoms — analogous to mjlab's terrain-body mesh extraction.

### Rendering

Rendering lives in `motrixsim.render`, separate from physics:

| Class / API | Purpose | Example |
|-------------|---------|---------|
| `RenderApp()` | Interactive window | Most control/ examples |
| `RenderApp(headless=True)` | Offscreen capture | `viewer/headless.py` |
| `RenderSettings(...)` | SSAO, SSGI, shadows, mesh simplify | `robot_locomotion.py` |
| `render.launch(model, batch=, render_offset=, render_settings=)` | Create render instances |
| `render.sync(data, wait=True/False)` | Push state to GPU renderer |
| `render.set_main_camera(camera)` | Follow / free camera |
| `render.gizmos.draw_sphere/arrow` | Debug draw | `site_and_sensor.py` |
| `camera.set_render_target("image", w, h)` | Per-camera RGB/depth | `go1.py` |
| `renderer.system_camera.capture()` | Headless RGB capture | `viewer/headless.py` |
| `motrixsim.viewer.launch(model, data)` | Lightweight viewer (no RenderApp loop) | `viewer/interactive_viewer.py` |

Decimation pattern used in locomotion examples:

```python
phys_dt = model.options.timestep
n_ctrl   = round(control_dt / phys_dt)    # policy rate
n_render = round(render_dt / phys_dt)     # 60 fps

# inner loop: model.step(data) every substep
# every n_ctrl: apply actions
# every n_render: render.sync(data)
```

### Useful robot / control examples

| File | Relevance to active-adaptation |
|------|--------------------------------|
| `control/robot_locomotion.py` | MSD scene build, terrain, keyboard commands, control/render decimation |
| `control/go1.py` | Observations (linvel, gyro, gravity, dof pos/vel), ONNX policy, fall reset |
| `go1_multi_task.py` | Reset API, joint init, multi-mode policies, termination checks |
| `utils/robot.py` | Thin state accessor pattern (good template for `MotrixEntity`) |
| `utils/policy.py` | ONNX locomotion policies for Go1/Go2/G1 |
| `physics/combine_msd.py` | Multi-robot scene composition |

### Mapping MotrixSim → active-adaptation adapters

Planned adapter responsibilities (mirroring mjlab):

#### `MotrixSimAdapter` (wraps `SceneModel` + optional `RenderApp`)

| Protocol method | MotrixSim implementation |
|-----------------|--------------------------|
| `get_physics_dt()` | `model.options.timestep` |
| `has_gui()` | `render is not None and not headless` |
| `step(render=False)` | `model.step(data)` — data held by scene adapter |
| `render()` | `render.sync(data)` |
| `set_camera_view(eye, target)` | `render.system_camera.set_view(...)` or attach a follower camera via MSD |

#### `MotrixSceneAdapter` (wraps `SceneData` + entity wrappers)

| Protocol member | MotrixSim implementation |
|-----------------|--------------------------|
| `num_envs` | `data.batch_size` or 1 if unbatched |
| `reset(env_ids)` | Partial reset via masked `SceneData` views; full reset via `SceneData(model, batch=...)` |
| `write_data_to_sim()` | Torch → numpy: actuator ctrls, external wrenches, root/joint state |
| `update(dt)` | Numpy → torch: dof pos/vel, link poses, sensor values into `MotrixEntityData` |
| `articulations` | `{"robot": MotrixEntity(...)}` wrapping `model.get_body(...)` |
| `sensors` | Named sensor handles or lazy `get_sensor_value` wrappers |
| `env_origins` | Tensor of per-env spawn offsets (may need explicit root-pose management) |
| `zero_external_wrenches()` | Clear per-link external force/torque buffers before each substep |
| `ground_mesh` | Warp mesh from hfield / floor geom |
| `create_sphere_marker` / `create_arrow_marker` | `render.gizmos` when GUI active; no-op headless |

#### `MotrixEntity` / `MotrixEntityData` (stub started in `adapter.py`)

Should expose the same torch tensors MDP code expects from mjlab `Entity.data`:

- `joint_pos`, `joint_vel`, `root_state` (pos + quat + lin/ang vel)
- `write_root_state_to_sim`, `write_joint_state_to_sim`
- `joint_names`, `body_names`

Bridge layer: numpy `SceneData` ↔ torch on `self.device`. Unlike mjlab, MotrixSim examples use numpy and CPU ONNX — device strategy TBD (likely CPU sim + optional GPU for policy).

### Asset configuration (`AssetCfg.motrix()`)

Each robot asset needs a motrix converter returning enough info to build an MSD scene:

- Path to MJCF (or inline MJCF fragment)
- Base link / body name
- Joint and body name lists (already in files like `assets/quadrupeds/a2.py`)
- Default joint positions, actuator names
- Sensor names for contact / IMU

Scene composition in `MotrixBackendEnv.setup_scene()` will likely mirror `robot_locomotion.build_model()`:

1. Load terrain scene via `msd.from_file`
2. Attach robot via `msd.from_file(asset_cfg.mjcf_path)`
3. Optionally attach cameras, objects
4. `model = scene.build()`
5. `data = SceneData(model, batch=(num_envs,))`

### MDP component compatibility

Start with `supported_backends = ("isaac", "mjlab", "motrix")` on components that only need generic articulation state. Components relying on Isaac-specific sensors, USD debug draw, or mjlab Viser viewer will stay Isaac/mjlab-only until ported.

---

## Motrix backend implementation plan

Step-by-step build order (each step should be testable in isolation):

1. **Fix imports** — use `motrixsim`, not `motrix`, in `adapter.py`.
2. **`MotrixSimAdapter`** — wrap `SceneModel`; implement `step` / `get_physics_dt`; optional headless (no `RenderApp`).
3. **`MotrixEntityData` + `MotrixEntity`** — numpy↔torch buffers for one articulation; read/write joint and root state via Body/Link/Joint APIs.
4. **`MotrixSceneAdapter`** — batch `SceneData`, `articulations` dict, `write_data_to_sim` / `update` / `reset(env_ids)`.
5. **`MotrixBackendEnv.setup_scene()`** — load asset via registry + MSD, create batched `SceneData`, wire adapters.
6. **`AssetCfg.motrix()`** — per-robot MJCF path + metadata (start with one robot, e.g. Go2 or A2).
7. **Register backend** — `aa.set_backend("motrix")`, `make_env_policy()` branch, `cfg/train.yaml` option.
8. **Terrain + spawn** — plane first; rough/hfield later via `TerrainScanner` or hfield assets.
9. **Viewer** — optional `RenderApp` when `headless=False`; debug gizmos for markers.
10. **MDP parity** — enable observations/rewards/actions one term at a time; annotate `supported_backends`.

Smoke test target (`motrix/test.py`):

```python
model = mx.load_model(...)
data = mx.SceneData(model, batch=(4,))
sim = MotrixSimAdapter(model, data)
scene = MotrixSceneAdapter(model, data, ...)
for _ in range(100):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
```

---

## Key design decisions

- **Not runtime-hot-swappable**: backend is set once per process at `aa.init()` time; asset imports depend on it
- **Protocol-based (not ABC)**: adapters use structural typing, so native objects can be wrapped without subclassing
- **Feature parity is intentionally incomplete**: mujoco is debug-only; many MDP terms are Isaac/mjlab only; motrix will start with a minimal subset
- **Registry pattern throughout**: env classes, MDP components, assets, and terrains all use `RegistryMixin` for lazy, backend-filtered registration
- **MotrixSim batch spacing**: physics instances share origin unless root poses are written explicitly — unlike Isaac/mjlab `env_spacing`
