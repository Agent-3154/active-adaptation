---
name: asset-definition
description: Define and register cross-backend robot/object assets in active-adaptation (Isaac Lab ArticulationCfg + mjlab EntityCfg via AssetSpec). Use when adding or editing files under assets/, wiring robot.name in cfg/task/, setting joint_names_simulation / body_names_simulation, contact sensors, actuators, URDF mimic / MJCF equality constraints, symmetry mappings, AssetSpec wrappers, mjlab spec_fn/CollisionCfg/ContactMatch, floating props, or cleaning up outdated mujoco-backend / in-repo MJCF patterns.
---

# Asset definition (active-adaptation)

Define robots and scene objects so **Isaac** and **mjlab** share one registry entry, the same init/symmetry metadata, and a **canonical joint/body order** for MDP tensors.

**Supported backends for new work:** `isaac` / `isaaclab` and `mjlab` only. The standalone `mujoco` backend is deprecated (mjlab replaces it) — do not add `MJArticulationCfg` or mujoco-only factories.

**Primary references**
- Shared cfg + `AssetSpec`: `active_adaptation/assets/asset_cfg.py`
- Canonical robot example: `active_adaptation/assets/quadrupeds/a2.py`
- Humanoid (multi-actuator): `active_adaptation/assets/humanoids/g1.py`
- Wrapper pattern: `active_adaptation/assets/underwater/BlueROV.py` + `envs/robots/TEACHME.md`
- Backend consumers: `envs/backends/isaac/env.py`, `envs/backends/mjlab/env.py`
- Model cache: `ROBOT_MODEL_DIR` → `<repo>/.cache/aa-robot-models/` (HF `btx0424/aa-robot-models`)
- **mjlab source of truth** (sibling repo): `mjlab/src/mjlab/{entity,actuator,sensor,scene,utils/spec_config}.py`; zoo examples `mjlab/asset_zoo/robots/*/`; docs `mjlab/docs/source/{entity,actuators,sensors,scene}.rst`

Read [reference.md](reference.md) for file map, **mjlab API contracts**, outdated cleanup, and templates.

**Related skills:** `environment-mdp` (uses `*_names_simulation`), `onpolicy-algorithms` (symmetry / symaug).

---

## When to use

- Adding a new robot or object under `active_adaptation/assets/`
- Porting an asset so it runs on both Isaac and mjlab
- Fixing policy transfer bugs caused by joint/body order mismatch
- Wiring `robot.name` in `cfg/task/**/*.yaml`
- Configuring coupled / mimic joints (URDF `<mimic>` ↔ MJCF `<equality>`)
- Cleaning outdated paths (`assets/Go2/`, `assets/G1/` vendored MJCF, `spawn.py`, mujoco backend branch)

---

## Hard rules

1. **Return `AssetSpec`** for articulated robots — `config` + optional `sensors` + optional `wrapper`. Do not hand raw Isaac/mjlab cfgs to the robot slot.
2. **Two factories + dispatcher** — `make_isaaclab_cfg()`, `make_mjlab_cfg()`, `make_cfg(backend: Literal["isaaclab", "mjlab"])`. Backends call with `"isaaclab"` or `"mjlab"` (not `"isaac"`).
3. **Always set simulation order** — `joint_names_simulation` and `body_names_simulation` on both backend cfgs (same lists). MDP terms resolve against these via `find_joints` / `find_bodies`.
4. **Share cross-backend constants** — `INIT_POS`, `INIT_JOINT_POS`, symmetry maps, effort/stiffness/damping, and the simulation name lists live at module top; only spawn/spec/actuator *types* differ per backend.
5. **Models live in `ROBOT_MODEL_DIR`** — USD + MJCF under `.cache/aa-robot-models/<robot>/`. Do not vendor large meshes into `assets/` for new robots.
6. **Register + import** — `registry.register("asset", "<name>", make_cfg)` and import the module from the package `__init__.py` so registration runs.
7. **Name parity** — joint/body names used by MDP, init regexes, and sensors must match across USD and MJCF (order may differ; the simulation lists fix layout). We always assume the USD joint and body names match those of the MJCF (if provided). So do not bother checking them.
8. **No new mujoco-backend assets** — ignore `elif aa.get_backend() == "mujoco"` in `asset_cfg.py` for new work; prefer deleting it when cleaning.
9. **mjlab actuators: `BuiltinPdActuatorCfg` by default** — AA joint actions often call both `set_joint_position_target` and `set_joint_velocity_target` (`envs/mdp/actions/joint.py`). Use `BuiltinPositionActuatorCfg` only when explicitly specified.
10. **Mimic / coupled joints: physics constraints, drivers only** — Isaac: URDF `<mimic>`; mjlab: MJCF `<equality><joint …/></equality>` (no URDF-style mimic tag). Actuate **driver** joints only; leave mimics unactuated (or Isaac zero-gain / passive). Prefer physics coupling over software target-copy (`MimicJointPosition`) for mjlab. Details: [reference.md](reference.md#mimic--coupled-joints).
11. **Never smoke-test with the shared root venv** — it is by design incomplete. Use `uv run --project venv/isaac51` and `uv run --project venv/mjlab`. See [.agents/skills/README.md](../README.md#smoke-tests--running-code).

---

## Checklist: new robot

```
Task Progress:
- [ ] Place USD + MJCF (and meshes) under ROBOT_MODEL_DIR / <robot>/
- [ ] Create assets/<family>/<robot>.py with shared INIT_*, JOINT/BODY_NAMES_SIMULATION, symmetry
- [ ] make_isaaclab_cfg → ArticulationCfg (UsdFileCfg) + ContactSensorCfg dict
- [ ] make_mjlab_cfg → EntityCfg (spec_fn → MjSpec) + ContactSensorCfg tuple
- [ ] Both cfgs get the same joint_names_simulation / body_names_simulation / symmetry
- [ ] Match actuator gains (stiffness, damping, effort, armature/friction) across backends
- [ ] If mimic joints: URDF `<mimic>` (Isaac) + matching MJCF `<equality>` (mjlab); actuators cover drivers only
- [ ] Action YAML `action_scaling` lists **drivers only** (mimics follow physics / optional Isaac `MimicJointPosition`)
- [ ] make_cfg(backend) dispatcher; register under a stable name
- [ ] Import module from assets/<family>/__init__.py
- [ ] Set robot.name in cfg/task YAML
- [ ] Smoke via backend venvs (`venv/isaac51` + `venv/mjlab`): joint/body counts and contact sensor
```

---

## Architecture

```
cfg/task/*.yaml  robot.name: unitree_a2
        │
        ▼
registry.get("asset", name) → make_cfg(backend=...)
        │
        ├── backend="isaaclab" → make_isaaclab_cfg() → AssetSpec
        │         ArticulationCfg + sensors: dict[str, ContactSensorCfg]
        │
        └── backend="mjlab"    → make_mjlab_cfg()    → AssetSpec
                  EntityCfg + sensors: tuple[ContactSensorCfg, ...]
        │
        ▼
IsaacBackendEnv / MjLabBackendEnv  → scene.robot / entities["robot"]
```

`AssetSpec` fields:

| Field | Role |
|-------|------|
| `config` | Backend articulation/entity cfg |
| `sensors` | Isaac: **dict** name→cfg; mjlab: **tuple** of named cfgs |
| `wrapper` | Optional instance (e.g. `UnderwaterRobot`); backend calls `_initialize` + lifecycle hooks |

---

## Shared metadata (required)

Define once at module scope (see `quadrupeds/a2.py`):

| Constant | Purpose |
|----------|---------|
| `INIT_POS` / `INIT_JOINT_POS` | Spawn pose; joint_pos may use regex keys |
| `JOINT_NAMES_SIMULATION` | Canonical joint order for obs/actions |
| `BODY_NAMES_SIMULATION` | Canonical body order for rewards/contacts |
| `JOINT_SYMMETRY_MAPPING` | Left/right joint pairs for symaug (`mirrored({...})`) |
| `SPATIAL_SYMMETRY_MAPPING` | Left/right body pairs |

Helpers in `asset_cfg.py`: `to_simulation_joint_order`, `to_simulation_body_order`, `sort_names_by_preferred_order`.

---

## Backend factories (what differs)

### Isaac (`make_isaaclab_cfg`)

- Import from `active_adaptation.assets.asset_cfg`: `ArticulationCfg`, `ImplicitActuatorCfg`, `sim_utils`, `AssetSpec`
- `spawn=sim_utils.UsdFileCfg(usd_path=..., activate_contact_sensors=True, ...)`
- `actuators={...: ImplicitActuatorCfg(joint_names_expr=..., effort_limit_sim=..., stiffness=..., damping=..., armature=..., friction=...)}`
- Split **driven** vs **mimic** groups when the URDF has `<mimic>`: drivers get real PD gains; mimics get zero stiffness/damping (`hand_passive`-style) so PhysX coupling owns them — see [Mimic joints](#mimic--coupled-joints)
- Sensors as **dict**, typically:

```python
sensors = {
    "contact_forces": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        track_air_time=True,
        history_length=3,
    )
}
```

### mjlab (`make_mjlab_cfg`)

- Import AA-extended `EntityCfg` from `asset_cfg` (adds `*_names_simulation` / symmetry); mjlab actuators / `CollisionCfg` / `ContactSensorCfg` / `ContactMatch`
- `spec_fn` → `mujoco.MjSpec` via `from_file` / `from_string` / procedural API (lazy-import `mujoco` inside the factory)
- **≤1 freejoint per entity** — each floating body is its own `SceneCfg.entities` entry
- `articulation=None` for passive props; robots need `EntityArticulationInfoCfg(actuators=(...))`
- Actuators (see [reference.md](reference.md#mjlab-api)):
  - **Default: `BuiltinPdActuatorCfg`** — paired `<position>`+`<velocity>`; required when actions set both pos and vel targets (`JointReferenceModel`, `LeakyVelocity`, `JointPosition`+FF, etc. in `envs/mdp/actions/joint.py`)
  - `BuiltinPositionActuatorCfg` — only if explicitly specified (`<position>` only; vel target ignored / implied 0)
  - Match with `target_names_expr` regex tuple; wrong transmission namespace → hard error
  - Mimic DOFs: **omit** from actuators (equality in MJCF drives them); do not PD-actuate intermediate/distal fingers, etc.
- `collisions=(CollisionCfg(...),)` — **`disable_other_geoms=True` by default** (zeros contype/conaffinity on non-matches). Name collision geoms; match them explicitly (e.g. `.*_collision`)
- **mjlab ≥1.6 structural fields are required:** always set `contype`, `conaffinity`, `condim`, and `priority`. Dict values for those fields must cover every matched geom (add a catch-all `".*"` / `".*_collision.*"`). Tuning fields (`friction`, `solref`, …) may stay partial / `None` (inherit XML).
- AA’s `mjlab/env.py` wraps `CollisionCfg.edit_spec` with an **empty-match fail-fast** (stock mjlab still silently disables all geoms when the regex matches nothing).
- Sensors as **tuple on AssetSpec**, not on EntityCfg — live on `SceneCfg.sensors`. Include every MDP-read quantity in `fields` (typically `("found", "force")`); Isaac exposes `net_forces_w` always, mjlab only allocates listed fields — see [environment-mdp](../environment-mdp/reference.md#contact-sensor-data-fields-isaac--mjlab).

```python
ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(mode="body", pattern=".*", entity="robot"),  # regex scoped to entity
    secondary=ContactMatch(mode="body", pattern="terrain", entity=None),  # literal global name
    fields=("found", "force"), reduce="netforce",  # or "maxforce"
    num_slots=1, track_air_time=True, history_length=3,
)
```

`ContactMatch.entity` set → pattern is regex on that entity’s unprefixed names. `entity=None` → pattern is a **literal** MuJoCo name (plane terrain body is `"terrain"`).

### Mimic / coupled joints

URDF and MJCF express coupling differently. Prefer **physics** constraints on both backends; keep mimic joints in `JOINT_NAMES_SIMULATION` (they are still DOFs) but **exclude them from action spaces**.

| Backend | Model file | Coupling | Actuators |
|---------|------------|----------|-----------|
| Isaac | URDF / USD from URDF | `<mimic joint="driver" multiplier="m" offset="o"/>` on the child joint | Drivers: normal PD. Mimics: zero-gain / passive group (or rely on PhysX mimic alone) |
| mjlab | MJCF | `<equality><joint joint1="driver" joint2="mimic" polycoef="o m 0 0 0"/></equality>` | Drivers only — no `BuiltinPdActuatorCfg` on mimic joints |

**polycoef:** MuJoCo enforces `q2 = c0 + c1*q1 + …` with `joint1` = independent (driver), `joint2` = dependent (mimic). Linear URDF mimic → `"offset multiplier 0 0 0"`.

**Actions / MDP:** `action_scaling` and `JointPosition` must match **drivers only**. Software `MimicJointPosition` (copies targets) is an Isaac fallback / shared path — do **not** use it as a substitute for missing MJCF equality on mjlab.

**Examples:** mjlab YAM gripper (`i2rt_yam` equality + actuate one finger); HOI G1 Inspire (`object_hoi` `g1_*_inspire_hand_DFQ.xml` equality + driver-only hand actuators). Full recipes: [reference.md](reference.md#mimic--coupled-joints).

### Floating objects (mjlab)

```python
# articulation=None (default); one freejoint; named collision geom
EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(...), rot=(1, 0, 0, 0)),
    spec_fn=...,  # freejoint + mesh/box geom
    collisions=(CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=1, conaffinity=1, condim=3, priority=0,
    ),),
)
```

Wire via task `objects:` → `mjlab/env.py` merges into `SceneCfg.entities`. Isaac returns `RigidObjectCfg`; mjlab returns `EntityCfg` (not required to wrap in `AssetSpec`).

Do not add `"mujoco"` branches. Optional `"motrix"` via `motrix_mjcf_path_fn` — omit unless that stack is in use.

---

## Registration and task wiring

1. Module side-effect: `registry.register("asset", "<name>", make_cfg)`
2. Package import: e.g. `assets/quadrupeds/__init__.py` imports the submodule
3. Task YAML:

```yaml
robot:
  name: unitree_a2
```

Both backends support `objects:` entries via `registry.get("asset", _target_)`. Factories may return raw `RigidObjectCfg` (Isaac) or `EntityCfg` (mjlab) rather than `AssetSpec` — see `dummy_objects.py` and project `hoi_object`.

### Dispatcher

```python
def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    if backend == "mjlab":
        return make_mjlab_cfg()
    raise ValueError(f"Invalid backend: {backend}")

registry.register("asset", "unitree_a2", make_cfg)
```

---

## mjlab hard rules (from upstream)

1. **≤1 freejoint / entity** — extra free bodies → separate `SceneCfg.entities` keys.
2. **Fixed-base auto-mocap** — no freejoint ⇒ `auto_wrap_fixed_base_mocap`; place via reset, else stuck at origin.
3. **`joint_pos=None`** requires an MJCF keyframe; otherwise use a regex dict (default `{".*": 0.0}`).
4. **Freejoint qpos** = `[pos(3), quat_wxyz(4), ...joint_qpos]`.
5. **Sensors on SceneCfg**, not EntityCfg; `ContactMatch.entity` must be an entities key (or `None` for literal names).
6. **`CollisionCfg.disable_other_geoms` defaults True** — unmatched geoms lose collision; name + match collision geoms deliberately. Empty match would wipe all collisions; AA raises in `mjlab/env.py`.
7. **`CollisionCfg` structural fields required (mjlab ≥1.6)** — pass `contype`, `conaffinity`, `condim`, `priority`; dict patterns need a catch-all for every matched geom.
8. **Name geoms** used by CollisionCfg / contact sensors (unnamed geoms are hard to match).
9. Prefer **`BuiltinPdActuatorCfg`** for robots (AA actions often set pos **and** vel targets via `envs/mdp/actions/joint.py`). Use `BuiltinPositionActuatorCfg` only when explicitly specified.
10. Prefer **fresh EntityCfg per call** (zoo `get_*_robot_cfg()` pattern) to avoid mutation.
11. Entity-level `<option>` does **not** propagate through scene attach — use sim `MujocoCfg`.
12. Import `mujoco` / mjlab types **inside** `make_mjlab_cfg` so Isaac-only processes do not import them.
13. **Mimic joints** → `<equality>` in MJCF + actuate drivers only (see [Mimic / coupled joints](#mimic--coupled-joints)). Unactuated joints are allowed (YAM / G1 Inspire pattern).

---

## Variants and wrappers

- **Composition:** extend base lists (see `a2_manipulator.py` appending arm joints/bodies onto A2 constants).
- **`AssetSpec.wrapper`:** instance with config-only `__init__`; backend `_initialize(robot=..., env=...)` then registers lifecycle hooks. See underwater TEACHME.
- **Isaac-only / mjlab-only:** raise `NotImplementedError` in the unsupported factory (e.g. BlueROV mjlab) rather than registering a broken cfg.

---

## Outdated patterns (cleanup targets)

When editing assets, prefer migrating away from:

| Pattern | Prefer |
|---------|--------|
| `asset_cfg` `mujoco` / `MJArticulationCfg` branch | Isaac + mjlab only |
| In-repo `assets/Go2/`, `assets/G1/**` MJCF/URDF as source of truth | `ROBOT_MODEL_DIR` (g1 still points at `assets/G1` — migrate when touching) |
| `assets/spawn.py` custom cloner | Standard Isaac Lab spawn / scene; not part of AssetSpec |
| Factories returning bare cfg for robots | Always `AssetSpec` |
| mjlab factory with `sensors=()` while Isaac has contact | Add matching contact sensor |
| `go2.py` / `b2.py` / `g1.py` / manipulators using `BuiltinPositionActuatorCfg` | Migrate to `BuiltinPdActuatorCfg` (AA actions often set vel targets) |
| Divergent actuator gains across backends | Keep stiffness/damping/effort aligned |
| MJCF with free mimic DOFs / PD on intermediates | `<equality>` + driver-only actuators (match URDF `<mimic>`) |
| Relying on `MimicJointPosition` alone for mjlab | Physics equality in the MJCF |

Full notes: [reference.md](reference.md#outdated-and-cleanup).

---

## Anti-patterns

- Accessing the live scene / articulation in the factory (factories build **cfg** only)
- Omitting `joint_names_simulation` / `body_names_simulation`
- Different name lists or different init regex coverage between Isaac and mjlab
- Using backend-native joint order in MDP (breaks transfer) — see `environment-mdp`
- Committing large USD/MJCF/meshes into `assets/` instead of the HF cache layout
- Registering without importing the module
- Passing `backend="isaac"` (must be `"isaaclab"`)
- New assets targeting the deprecated standalone mujoco backend
- Multiple freejoints in one EntityCfg / one free body spanning robot+object
- Putting ContactSensorCfg on EntityCfg (must be AssetSpec.sensors → SceneCfg.sensors)
- Relying on `CollisionCfg` default without matching collision geoms (everything else disabled)
- Omitting `contype` / `conaffinity` / `condim` / `priority`, or using a structural dict without a catch-all (mjlab ≥1.6 raises)
- `ContactMatch(entity=None, pattern=...)` expecting regex (literal only when entity unset)
- Using `BuiltinPositionActuatorCfg` for robots whose actions call `set_joint_velocity_target` (vel targets are dropped)
- Actuating mimic joints on mjlab while also declaring equality (fighting the constraint)
- Putting mimic joints in task `action_scaling` / policy action dim
- Assuming MJCF inherits URDF `<mimic>` (it does not — add `<equality>` explicitly)
- Inspecting USD (usdcat / pxr / string dumps) to verify joint/body names against MJCF — trust the MJCF names

---

## Minimal skeleton

```python
from typing import Literal
from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored

registry = Registry.instance()

INIT_POS = (0.0, 0.0, 0.5)
INIT_JOINT_POS = {".*_joint": 0.0}
JOINT_SYMMETRY_MAPPING = mirrored({...})
SPATIAL_SYMMETRY_MAPPING = mirrored({...})
JOINT_NAMES_SIMULATION = [...]
BODY_NAMES_SIMULATION = [...]

def make_isaaclab_cfg():
    from isaaclab.sensors import ContactSensorCfg
    from active_adaptation.assets.asset_cfg import (
        AssetSpec, ArticulationCfg, ImplicitActuatorCfg, sim_utils,
    )
    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_MODEL_DIR / "my_robot" / "robot.usd"),
            activate_contact_sensors=True,
            # rigid_props / articulation_props / collision_props ...
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS, joint_pos=INIT_JOINT_POS, joint_vel={".*": 0.0},
        ),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=..., stiffness=..., damping=...,
            armature=0.01, friction=0.01,
        )},
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = {
        "contact_forces": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            track_air_time=True, history_length=3,
        )
    }
    return AssetSpec(config=cfg, sensors=sensors)

def make_mjlab_cfg():
    import mujoco
    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg
    from mjlab.actuator import BuiltinPdActuatorCfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch

    path = ROBOT_MODEL_DIR / "my_robot" / "robot.xml"
    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS, joint_pos=INIT_JOINT_POS, joint_vel={".*": 0.0},
        ),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(path)),
        articulation=EntityArticulationInfoCfg(
            actuators=(BuiltinPdActuatorCfg(
                target_names_expr=(".*",),
                effort_limit=..., stiffness=..., damping=...,
                armature=0.01, frictionloss=0.01,
            ),),
        ),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision",),
            contype=0, conaffinity=1,  # no self-collision among matched geoms
            condim=3, priority=0,      # required structural fields (mjlab ≥1.6)
            # disable_other_geoms=True by default
        ),),
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
            fields=("found", "force"), reduce="netforce",
            num_slots=1, track_air_time=True, history_length=3,
        ),
    )
    return AssetSpec(config=cfg, sensors=sensors)

def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    if backend == "mjlab":
        return make_mjlab_cfg()
    raise ValueError(f"Invalid backend: {backend}")

registry.register("asset", "my_robot", make_cfg)
```
