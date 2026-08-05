# Asset definition — reference

Companion to [SKILL.md](SKILL.md). File map, backend wiring, templates, and cleanup inventory.

---

## File map

| Path | Role |
|------|------|
| `active_adaptation/assets/asset_cfg.py` | Backend-gated `ArticulationCfg` (Isaac) / `EntityCfg` (mjlab); `AssetSpec`; order helpers |
| `active_adaptation/assets/__init__.py` | Imports family packages (registration side effects) |
| `active_adaptation/assets/quadrupeds/*.py` | Unitree A2/B2/Go2 (+ manipulators) |
| `active_adaptation/assets/humanoids/g1.py` | G1 waist-unlocked |
| `active_adaptation/assets/underwater/*.py` | BlueROV (+ Heavy); Isaac + `AssetSpec.wrapper` |
| `active_adaptation/assets/dummy_objects.py` | Isaac-only rigid objects (returns bare cfg) |
| `active_adaptation/assets/spawn.py` | Legacy Isaac cloner decorator — **not** used by AssetSpec robots |
| `active_adaptation/assets/Go2/`, `G1/` | In-repo MJCF/URDF leftovers — prefer `ROBOT_MODEL_DIR` |
| `active_adaptation/__init__.py` | `ROBOT_MODEL_DIR = CACHE_DIR / "aa-robot-models"` |
| `envs/backends/isaac/env.py` | `asset_factory(backend="isaaclab")`; sensors dict → scene |
| `envs/backends/mjlab/env.py` | `asset_factory(backend="mjlab")`; sensors tuple → SceneCfg |
| `envs/utils/api.py` | `find_joints` / `find_bodies` against `*_names_simulation` |
| `cfg/task/**/*.yaml` | `robot.name` selects registry key |

---

## `asset_cfg.py` behavior

At import time, `aa.get_backend()` selects which cfg class exists:

| Backend | Exported robot cfg | Notes |
|---------|-------------------|--------|
| `isaac` | `ArticulationCfg` extends Isaac Lab | Also re-exports `ImplicitActuatorCfg`, `sim_utils` |
| `mjlab` / `motrix` | `EntityCfg` extends mjlab `EntityCfg` | Optional `motrix_mjcf_path_fn` |
| `mujoco` | imports `MJArticulationCfg` | **Deprecated** — do not extend |

Extra fields on both Isaac/mjlab cfgs:

- `joint_symmetry_mapping`
- `spatial_symmetry_mapping`
- `joint_names_simulation`
- `body_names_simulation`

`AssetSpec.with_wrapper(wrapper)` mutates and returns `self` for chaining.

---

## Backend consumption

### Isaac

```python
asset_spec = asset_factory(backend="isaaclab")
scene_cfg.robot = asset_spec.config
for name, sensor_cfg in asset_spec.sensors.items():
    setattr(scene_cfg, name, sensor_cfg)
scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
# if asset_spec.wrapper: wrapper._initialize(robot=..., env=...); register hooks
```

Objects (optional):

```yaml
objects:
  stand:
    _target_: dummy_stand
  object:
    _target_: hoi_object
    motion_path: ...
```

| Backend | Factory return | Scene wiring |
|---------|----------------|--------------|
| Isaac | `RigidObjectCfg` / `ArticulationCfg` (not necessarily `AssetSpec`) | `isaac/env.py` → `scene_cfg.<name>` |
| mjlab | `EntityCfg` (or `AssetSpec.config`) | `mjlab/env.py` → `SceneCfg.entities[name]` |

Floating props: freejoint + mesh geoms; see `object_hoi` `hoi1/assets/objects.py` (STL / optional MJCF).

### mjlab

```python
asset_spec = asset_factory(backend="mjlab")
entities = {"robot": asset_spec.config}
for obj_name, obj_spec in cfg.get("objects", {}).items():
    obj_spec = dict(obj_spec)
    fn = registry.get("asset", obj_spec.pop("_target_"))
    obj_cfg = fn(backend="mjlab", **obj_spec)
    entities[obj_name] = obj_cfg.config if isinstance(obj_cfg, AssetSpec) else obj_cfg
scene_cfg = SceneCfg(
    entities=entities,
    sensors=tuple(asset_spec.sensors),
    terrain=...,
)
```

Contact sensor `entity="robot"` must match an `entities` key. Terrain secondary uses literal `pattern="terrain"` with `entity=None`.

---

## mjlab API

Upstream package lives at `/home/btx0424/lab51/mjlab` (or installed `mjlab`). Prefer these over outdated migration notes.

### Path map

| Area | Path |
|------|------|
| Entity / InitialState / ArticulationInfo | `mjlab/entity/entity.py` |
| VariantEntityCfg (per-world meshes) | `mjlab/entity/variants.py` |
| Spec editors (Collision/Light/Camera/…) | `mjlab/utils/spec_config.py` |
| `auto_wrap_fixed_base_mocap` | `mjlab/utils/spec.py` |
| Builtin Position / Pd / Motor | `mjlab/actuator/builtin_actuator.py` |
| XmlActuator | `mjlab/actuator/xml_actuator.py` |
| ContactSensor / ContactMatch | `mjlab/sensor/contact_sensor.py` |
| SceneCfg | `mjlab/scene/scene.py` |
| Zoo robots | `mjlab/asset_zoo/robots/{unitree_go1,unitree_g1,i2rt_yam}/*_constants.py` |
| Floating prop example | `mjlab/tasks/manipulation/config/yam/env_cfgs.py` (`get_cube_spec`) |
| Docs | `mjlab/docs/source/{entity,actuators,sensors,scene,faq}.rst` |

### Entity type matrix

| | Fixed (no freejoint) | Floating (1 freejoint) |
|--|----------------------|------------------------|
| Non-articulated | table/wall (auto-mocap) | box/ball/HOI object |
| Articulated | arm/door | humanoid/quad |

- `articulation=None` → no actuators (props / passive).
- `is_articulated` ↔ non-free joints exist; `is_actuated` ↔ actuators present.
- `VariantEntityCfg`: floating only; do **not** set `spec_fn`; use `variants` dict of mesh-differing specs.

### `spec_fn`

```python
# Preferred for robots (meshdir relative to XML)
mujoco.MjSpec.from_file(str(mjcf_path))  # <compiler meshdir="assets" angle="radian"/>

# Procedural prop (yam cube pattern)
spec = mujoco.MjSpec()
body = spec.worldbody.add_body(name="cube")
body.add_freejoint(name="cube_joint")
body.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(...), mass=...)
```

STL props: XML with `meshdir` + `<mesh file="..."/>`, or procedural `MjSpec.from_string` with absolute meshdir (see `object_hoi` `hoi1/assets/objects.py`).

### Actuators

| Class | MuJoCo elements | When |
|-------|-----------------|------|
| **`BuiltinPdActuatorCfg`** | `<position>` + `<velocity>` | **AA default.** Consumes both pos and vel targets from `set_joint_position_target` / `set_joint_velocity_target` (`envs/mdp/actions/joint.py`: `JointReferenceModel`, leaky velocity models, pos+FF, …). |
| `BuiltinPositionActuatorCfg` | `<position>` only (v≈0) | Only when explicitly specified; mjlab zoo default, **not** AA default. Vel targets from actions are ignored. |
| `XmlActuatorCfg` | wrap existing XML actuators | Keep MJCF actuators |

**Why Pd is default:** several AA action terms write non-zero velocity targets into the articulation. `BuiltinPositionActuatorCfg` only creates `<position>` actuators, so those vel commands have no effect and tracking/impedance differs from Isaac `ImplicitActuatorCfg`.

Shared: `target_names_expr`, `armature` / `frictionloss` / `viscous_damping` (`None` keeps XML), `transmission_type` (`JOINT`/`TENDON`/`SITE`). Position/Pd reject SITE.

Zoo pattern (for reference only): armature from reflected rotor inertia; stiffness = armature × ω²; multi-group disjoint regexes — still prefer wrapping those gains in `BuiltinPdActuatorCfg` inside AA.

### CollisionCfg

```python
CollisionCfg(
  geom_names_expr=(".*_collision",),  # required named geoms
  contype=0, conaffinity=1,           # collide with world, not each other
  condim={foot_re: 6, ".*_collision": 1},
  friction={foot_re: (1, 5e-3, 5e-4)},
  solref=(0.01, 1),
  disable_other_geoms=True,           # DEFAULT — disables non-matches
)
```

Feet-only (Go1): match feet only + `disable_other_geoms=True`. Full body: match all `*_collision`.

AA’s `mjlab/env.py` patches `CollisionCfg.edit_spec` to error if the matched subset is empty.

### ContactMatch / ContactSensorCfg

- Sensors belong on **`SceneCfg.sensors`**, not EntityCfg.
- `entity="robot"` → regex on that entity’s unprefixed names (scene prefixes as `robot/...`).
- `entity=None` → **literal** global MuJoCo name (`"terrain"` for plane terrain; terrain attaches **without** entity prefix).
- `fields`: `found`, `force`, `torque`, `dist`, `pos`, `normal`, `tangent`.
- `reduce`: `none` / `mindist` / `maxforce` / `netforce`.
- `track_air_time` requires `"found"` in fields. Use `history_length` ≈ control decimation for brief contacts.

### InitialStateCfg

```python
EntityCfg.InitialStateCfg(
  pos=(x, y, z),
  rot=(w, x, y, z),          # WXYZ
  lin_vel=..., ang_vel=...,  # floating root state (not keyframe qpos)
  joint_pos={".*": 0.0},     # None ⇒ require model keyframe
  joint_vel={".*": 0.0},
)
```

Freejoint keyframe qpos: `[pos(3), quat_wxyz(4), ...hinge_qpos]`.

### Scene attach

Order: terrain (no prefix) → entities with `{name}/` prefix → sensors. Keyframes merge in entity iteration order into scene `init_state`.

### Gotchas

| Symptom | Cause |
|---------|--------|
| Compile / attach fail | >1 freejoint in one entity; bad meshdir; missing mesh file |
| Entity build fail | `joint_pos=None` without keyframe; actuator regex empty / wrong transmission |
| Contact edit_spec fail | `entity` not in `entities`; primary matches nothing |
| No contacts | `disable_other_geoms` wiped geoms; contype/conaffinity miss terrain; visual-only geoms; secondary not literal `"terrain"` |
| Fixed robot at origin | mocap wrap without reset placement |
| Missed foot strikes | no `history_length` / `track_air_time` |

### AA bridge notes

- AA `asset_cfg.EntityCfg` subclasses mjlab’s and adds `joint_names_simulation`, `body_names_simulation`, symmetry maps, optional `motrix_mjcf_path_fn`.
- Robot factories return `AssetSpec(config=EntityCfg, sensors=tuple[ContactSensorCfg, ...])`.
- Object factories may return bare `EntityCfg` (HOI); env unwraps `AssetSpec` if present.
- Do not confuse `import mujoco` inside mjlab factories (correct) with the deprecated standalone `backend=="mujoco"` path.

---

## Canonical examples

| Robot | Registry key | Isaac | mjlab | Notes |
|-------|--------------|-------|-------|-------|
| A2 | `unitree_a2` | yes | yes | Best AA template; `BuiltinPdActuatorCfg` |
| A2+Piper | `unitree_a2_piper` | yes | yes | Extends A2 name lists; still on Position — migrate to Pd |
| B2 | `unitree_b2` | yes | yes | mjlab sensors empty; still on Position — migrate to Pd |
| B2+Z1 | (see file) | yes | yes | Typo filename `b2_manipulatior.py`; Position → Pd |
| Go2 | `unitree_go2` | **no** | yes | Add Isaac; Position → Pd |
| G1 | `g1_waist_unlocked` | yes | yes | Paths under `assets/G1/`; Position → Pd |
| BlueROV | `bluerov` / heavy | yes | raise | Wrapper hydrodynamics |
| mjlab Go1 zoo | n/a | — | upstream ref | Uses Position; **do not copy** for AA assets |
| HOI G1 Inspire | `g1_inspire_hand_urdf` | yes | yes | `BuiltinPdActuatorCfg` |
| HOI object | `hoi_object` | RigidObjectCfg | EntityCfg | STL/XML free body |

---

## Simulation name lists

Isaac PhysX and MuJoCo often enumerate joints/bodies in different orders. Policies must see one layout.

1. Author authoritative lists in the asset module.
2. Attach the **same** lists to both backend cfgs.
3. MDP uses `find_joints` / `find_bodies` / `find_sensor_bodies` (environment-mdp skill).

Lists must be a **permutation** of the names present in both USD and MJCF (for DOFs/bodies the MDP uses). Missing names break `sort_names_by_preferred_order` / index helpers.

Regex init maps (`INIT_JOINT_POS`) are resolved by each backend independently; ensure every actuated joint matches some key on both.

---

## Actuators

| Backend | Typical class | Limit / gain fields |
|---------|---------------|---------------------|
| Isaac | `ImplicitActuatorCfg` | `effort_limit_sim`, `velocity_limit_sim`, `stiffness`, `damping`, `armature`, `friction` |
| mjlab | **`BuiltinPdActuatorCfg` (AA default)** | `effort_limit`, `stiffness`, `damping`, `armature`, `frictionloss`, optional `viscous_damping` |

Keep numeric gains aligned across backends unless there is a documented sim-to-sim reason.

Multi-group actuators (G1): mirror the same joint-name partitions and gains in both factories.

See [mjlab API](#mjlab-api) for why Pd (not Position) is required when actions set velocity targets.

---

## Contact sensors

| | Isaac | mjlab |
|---|-------|-------|
| Container | `dict` keyed by scene attribute name | `tuple` of cfgs with `.name` on **SceneCfg** |
| Usual name | `contact_forces` | `name="contact_forces"` |
| Scope | `prim_path="{ENV_REGEX_NS}/Robot/.*"` | `primary` `entity="robot"` (regex); `secondary` literal `"terrain"` |
| Flags | `track_air_time`, `history_length` | same + `fields`, `reduce`, `num_slots` |

Enable `activate_contact_sensors=True` on Isaac `UsdFileCfg` when using contact sensors.

MDP contact indexing: use `find_sensor_bodies` (mjlab has no `contact_sensor.find_bodies`).

**Runtime data fields differ:** Isaac `sensor.data.net_forces_w` vs mjlab `sensor.data.force` (must list `"force"` in `ContactSensorCfg.fields`). Full table: [environment-mdp reference — Contact sensor data fields](../environment-mdp/reference.md#contact-sensor-data-fields-isaac--mjlab).

Full `ContactMatch` semantics: [mjlab API](#contactmatch--contactsensorcfg).

---

## Symmetry

```python
from active_adaptation.utils.symmetry import mirrored

JOINT_SYMMETRY_MAPPING = mirrored({
    "FL_hip_joint": (-1, "FR_hip_joint"),  # sign flip, partner name
    ...
})
SPATIAL_SYMMETRY_MAPPING = mirrored({
    "FL_foot": "FR_foot",
    "base_link": "base_link",
})
```

`mirrored` fills the reverse mapping. Required for PPO symaug; attach on both backend cfgs.

---

## Model assets (`ROBOT_MODEL_DIR`)

```
<repo>/.cache/aa-robot-models/
  a2/a2.usd  a2/a2.xml
  b2/...
  go2_unilab/go2.xml
  a2_piper/...
  underwater/BlueROV.usd
  scene/*.hdr
```

Source: Hugging Face dataset `btx0424/aa-robot-models`. See repo `README.md` § Asset download.

New robots: add files to that layout (and the HF dataset if shared); point factories at `ROBOT_MODEL_DIR / "<name>" / ...`.

---

## Outdated and cleanup

Use this when refactoring; do not expand these patterns.

### High priority

1. **`asset_cfg.py` mujoco branch** — remove once the mujoco backend package is deleted; new code must not import `MJArticulationCfg`.
2. **In-repo `assets/Go2/`, `assets/G1/`** — migrate `g1.py` paths to `ROBOT_MODEL_DIR`; delete duplicate MJCF/URDF when unused.
3. **`go2.py`** — add Isaac `make_isaaclab_cfg` + USD; switch mjlab to `BuiltinPdActuatorCfg`.
4. **`b2.py` / `g1.py` / manipulators** — replace `BuiltinPositionActuatorCfg` with `BuiltinPdActuatorCfg`; add contact sensors where missing.
5. **`spawn.py`** — Isaac-only experimental cloner; unrelated to AssetSpec; candidate for removal if unused.

### Medium

6. **`dummy_objects.py`** — Isaac-only; document or add mjlab; optionally wrap in `AssetSpec` for uniformity.
7. **`motrix` in `make_cfg`** — drop from new assets unless motrix is actively used; `EntityCfg.motrix_mjcf_path_fn` is mjlab-adjacent export, not a third robot backend.
8. **Filename typo** `b2_manipulatior.py` — rename carefully (imports + registry).
9. **Inconsistent effort field names on older Isaac cfgs** (`effort_limit` vs `effort_limit_sim`) — prefer `*_sim` fields as in A2/G1.

### Do not confuse

- Importing `mujoco` **inside** `make_mjlab_cfg` to call `MjSpec.from_file` is correct (mjlab stack).
- The deprecated path is the **standalone** `backend == "mujoco"` articulation cfg and any `envs/backends/mujoco` factories.

---

## Wrapper lifecycle (underwater)

1. Factory builds `UnderwaterRobot(...)` with hydro/rotor **config only**.
2. Isaac `setup_scene` takes `asset_spec.wrapper`, calls `_initialize(robot=self.robot, env=self)`.
3. Registers optional `startup` / `reset` / `pre_step` / `post_step` / `update` / `debug_draw`.
4. Wrapper applies external wrenches in `pre_step` / `write_data_to_sim`.

Details: `active_adaptation/envs/robots/TEACHME.md`.

---

## Smoke checks

After adding an asset:

1. Import path: `from active_adaptation.assets import …` (or start training) prints registry registration.
2. Isaac: create env with `robot.name`; assert `robot.joint_names` covers `joint_names_simulation`.
3. mjlab: same; assert contact sensor `primary_names` align via `find_sensor_bodies`.
4. Optional: one PPO/off-policy step on each backend; confirm obs dim identical.

---

## Related

- [environment-mdp](../environment-mdp/SKILL.md) — MDP terms and index helpers
- [onpolicy-algorithms](../onpolicy-algorithms/SKILL.md) — symmetry augmentation consumers
- mjlab docs: `docs/source/entity/index.rst`, `actuators.rst`, `sensors/index.rst`, `scene.rst`, `faq.rst`
