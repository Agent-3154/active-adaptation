# Environment / MDP — reference

## File map

```
active_adaptation/envs/
├── env_base.py              # _EnvBase: create / init / step / reset wiring
├── adapters.py              # SimAdapter, SceneAdapter, CameraFrustumHandle
├── backends/
│   ├── isaac/
│   │   ├── adapter.py       # Omni DebugDraw + Viser draw_*/frustum
│   │   ├── viewer.py        # IsaacViserViewer (meshes, lines, frustums)
│   │   └── env.py           # viewer.viser, clear_debug callback
│   └── mjlab/
│       ├── adapter.py       # Viser draw_*/frustum
│       ├── viewer.py        # MjLabViewer
│       └── env.py
└── mdp/
    ├── base.py              # MDPComponent, is_method_implemented
    ├── __init__.py          # re-exports V1+V2 bases + subpackages
    ├── actions/
    │   ├── base.py          # Action, ActionV2
    │   └── *.py             # e.g. JointPosition
    ├── commands/
    │   ├── base.py          # Command, CommandV2 (sync_state vs update)
    │   └── *.py             # e.g. Twist
    ├── observations/
    │   ├── base.py          # Observation, ObservationV2
    │   ├── common.py, joint.py, underwater.py, …
    │   └── __init__.py      # explicit submodule imports
    ├── rewards/
    │   ├── base.py          # Reward (deprecated), RewardV2
    │   ├── locomotion.py, …
    │   └── __init__.py      # auto-imports all *.py except base/common
    ├── terminations/
    ├── randomizations/
    └── …
```

Registry: `active_adaptation/registry.py` (`RegistryMixin.make`).

Task YAML: `cfg/task/<Robot>/<Task>.yaml` — blocks `input`, `command`, `observation`, `reward`, `termination`, `randomization`.

---

## Construction sequence

```
_EnvBase.__init__
  ├─ _create_mdp_terms()     # ActionV2/CommandV2/… .make from cfg; no scene yet
  ├─ _setup_simulation()     # setup_scene → sim/scene ready
  ├─ _initialize_mdp_terms() # term._initialize(self) for all terms
  └─ _build_tensor_specs()   # action/obs/reward/done specs from initialized terms
```

`ObsGroup` / `RewardGroup` call `_initialize` on each member, then probe `compute()` once to build shapes/specs.

---

## Step loop

```
_step(tensordict)
  ├─ for each input_manager: process_action(tensordict[key])
  ├─ for substep in decimation:
  │     zero external wrenches
  │     apply_action(substep)
  │     pre_step(substep)
  │     write_data_to_sim → sim.step → scene.update
  │     post_step(substep)
  ├─ episode_length_buf += 1
  ├─ command.sync_state()   # intermediates for rewards; do not change next-step targets
  ├─ update callbacks
  ├─ _compute_reward
  ├─ _compute_termination
  ├─ command.update()       # may resample; write next-step targets for obs
  ├─ _compute_observation
  └─ if sim.has_gui(): debug_draw callbacks
       (backends insert scene.clear_debug as first callback)
```

`sim.has_gui()` is true for: Isaac Omni Kit **or** `viewer.viser`; mjlab when a `MjLabViewer` exists (non-headless).

---

## Callback registration

| Source | How registered |
|--------|----------------|
| `command_manager` | Always: `pre_step`, `reset`, `debug_draw` |
| each `input_manager` | Always: `reset`, `debug_draw` |
| obs / reward / term / randomization | Via `_add_mdp_component`: only **overridden** methods among `startup`, `reset`, `pre_step`, `post_step`, `update`, `debug_draw` |

`update` callbacks are wrapped in `ScopedTimer(class_name)`.

Command `sync_state` / `update` are **not** in the generic callback lists; they are called explicitly in `_step`.

---

## Config parsing

```python
def parse_component_spec(name, cfg):
    kwargs = dict(cfg)
    target = kwargs.pop("_target_", name)
    return name, target, kwargs
```

- Reward groups: `_enabled_`, `_compile_` are group-level flags (popped before terms).
- Command: top-level `_target_` required.
- Observation: YAML key is the term name in the concatenated group; `_target_` selects the class when different.

---

## Registration / imports

Subclassing a V2 base registers `ClassName` in that base’s `registry`.

Rewards package auto-imports sibling modules:

```python
# rewards/__init__.py — importlib all *.py except base/common/_*
```

Observations use explicit `from . import common, joint, …`. New observation modules must be imported from `observations/__init__.py` (or another imported module).

Duplicate class names raise at import time with file:line of the conflict.

Optional: `namespace = "foo"` on the class → registry key `foo.ClassName`; YAML `_target_` must match.

---

## RewardV2 details

- `_compute` may return `Tensor` or `(rew, is_active)` (inactive envs zeroed for EMA count).
- `compute` applies `weight * rew * modifier`, then resets `modifier` to ones.
- Other terms can write into `reward.modifier` before compute for coupling.
- `relabel(tensordict)` exists for offline reward recomputation (see base).

---

## TerminationV2 details

- `compute(terminated)` receives the running terminated mask.
- Return bool tensor `(num_envs, 1)`, or `(term, discount)`.
- `is_timeout=True` → contributes to `truncated`; else `terminated`.
- `enabled=False` zeros the term contribution.

---

## ActionV2 details

- `_initialize` sets `self.asset = scene.articulations["robot"]`.
- Must define `action_dim` before specs are built.
- Optional `names` / `find_names` for joint subsets.
- `diagnostics()` optional dict for logging.

---

## CommandV2 details

- Abstract `sync_state` and `update` (must override both, even if `pass`).
- `sample_init(env_ids)` provides root (and optionally joint) state for `_reset_idx`.
- No teleop on V2 (legacy `Command` had `teleop`).
- **`sync_state`:** refresh intermediates for rewards from the *current* command; do not change commands / next-step targets.
- **`update`:** may change commands; write next-step targets for observations (rewards on the *next* step read these).
- **First obs discarded:** post-reset observation is invalid (`is_init`); do not recompute next-step targets in `reset` solely to validate it. See SKILL.md “Command timing”.

---

## Reset API

```python
# mdp/base.py
def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
    ...

# env_base._reset
# always passes a TensorDictBase (empty TD if caller passed None)
[callback(env_ids, tensordict) for callback in self._reset_callbacks]
```

Terms may read/write `tensordict`. Most leave it unused.

**Order:** `_reset_idx` (`sample_init`) → `scene.reset` → `reset` callbacks.

**Future:** drop `sample_init`; `reset` decides initial state.

---

## Raycasting / USD meshes

External package: [`simple-raycaster`](https://github.com/btx0424/simple-raycaster) (installed into the uv env with active-adaptation; not a workspace-local path). Public exports: `MultiMeshRaycaster`, `MultiMeshRaycasterV2`. Authoritative notes: upstream `AGENTS.md`.

### Package modules (import paths)

```
simple_raycaster.raycaster       # MultiMeshRaycaster — caller supplies mesh_pos_w / mesh_quat_w
simple_raycaster.raycaster_v2    # MultiMeshRaycasterV2 — Isaac entity poses auto-gathered
simple_raycaster.proximity       # MeshProximitySensor — closest-point / signed-distance queries
simple_raycaster.kernels         # Warp raycast + fused transform + proximity kernels
simple_raycaster.helpers         # trimesh2wp, quat_rotate_inverse, voxelize_*
simple_raycaster.utils_usd       # find_matching_prims, get_trimesh_from_prim, usd2trimesh, usd2wp
simple_raycaster.utils_mjc       # get_trimesh_from_body (MuJoCo)
```

### USD extraction pipeline (`utils_usd`)

```
stage / prim_path regex
  → find_matching_prims
  → get_mesh_prims_subtree (Mesh + Cube; resolve Usd instances → prototypes)
  → usd2trimesh per mesh prim
  → transform = mesh_world * parent_world⁻¹  (skip for prototype meshes)
  → concatenate + merge_vertices
  → body-local trimesh.Trimesh
```

- **Dynamic bodies:** mesh vertices stay in the body/parent frame; runtime pose is `body_link_pose_w` (`[pos(3), quat_wxyz(4)]`).
- **Static world geometry** (`add_isaac_static`): combine matching visuals once; bake world transform into the trimesh; raycast pose is identity.
- Prim path for Isaac articulations: template `entity.root_physx_view.prim_paths[0]`, then `{template with body_name}/visuals` per `entity.body_names`. Count must equal `entity.num_bodies`.

### Raycast call shape

- Inputs: `ray_starts_w`, `ray_dirs_w` as `[N, n_rays, 3]` (dirs normalized); optional `enabled [N]`, `mesh_indices [N, n_subset]`.
- Outputs: `hit_positions_w [N, n_rays, 3]`, `hit_distances [N, n_rays]` (closest hit across meshes; misses → `max_dist`).
- Prefer `raycast_fused` (single Warp kernel). Non-fused `raycast` is for parity/debug.
- Conventions: quats **WXYZ** in Python (fused kernel reorders to XYZW for Warp); `wp.init()` once; CUDA device for production.

### active-adaptation touchpoints

| Location | Role |
|----------|------|
| `IsaacSceneAdapter.ground_mesh` | Warp mesh for `/World/ground` (plane or USD) |
| `env.ground_mesh` | Proxy onto scene adapter |
| `observations/extero.raycast_camera` | **Canonical V2** usage: static ground + `add_isaac_entity` targets |
| `observations/extero.height_scan` | V1 + `ground_mesh` (+ optional `add_from_path` targets with manual root poses) |
| `observations/underwater` DVL | IsaacLab `raycast_mesh` on `ground_mesh` only (not multi-mesh V2 yet) |

---

## Debug visualization

Cross-backend API in `envs/adapters.py`. MDP terms call **`env.scene`**, never `env.debug_draw`.

### SceneAdapter surface

```python
scene.clear_debug()
scene.draw_vector(x, v, size=2.0, color=(…,))   # x,v: (..., 3)
scene.draw_point(x, color=(…,), size=10.0)
scene.draw_plot(x, size=2.0, color=(…,))         # polyline
handle = scene.create_camera_frustum(name, fov_y=…, aspect=…, scale=0.15)
handle.position = pos_w      # torch or numpy
handle.wxyz = quat_wxyz
handle.image = hwc_uint8
```

Env backends register `scene.clear_debug` as the **first** `debug_draw` callback so each frame starts empty, then term callbacks append primitives; viewers sync on `sim.step(render=True)` / `viewer.update()`.

### Backend behavior

| Backend | Primitives | Camera frustum | Enable |
|---------|------------|----------------|--------|
| Isaac Omni | `IsaacDebugDraw` (Kit) | — | Native GUI |
| Isaac Viser | `IsaacViserViewer` lines/points | `register_camera` → `CameraFrustumHandle` | `viewer.viser: true` |
| mjlab | `MjLabViewer` MDP buffers | same | non-headless |

Isaac adapter may fan out `draw_*` to **both** Omni and Viser when both exist. `IsaacSimAdapter.has_gui()` is true if either is present; Omniverse window setup must check the **native** sim (`sim._sim.has_gui()`), not the adapter.

### Viser mesh path (viewer internals)

Same extraction as raycasting (`utils_usd.get_trimesh_from_prim` / `{body}/visuals`):

1. Upload body meshes once at viewer `setup()`.
2. Each update: write `body_link_pose_w` into batched mesh handles (park non-selected envs far away rather than rebuilding handles).
3. Camera obs: frustum pose = body × mount offset (WXYZ); push RGB as HWC uint8.
4. Example term: `observations/underwater.uw_camera` with `debug_vis: true`.

### Example task knobs

```yaml
viewer:
  eye: [4.0, 4.0, 4.0]
  lookat: [0., 0., 0.]
  viser: true   # Isaac only; mjlab uses headless=false

observation:
  policy:
    underwater.uw_camera:
      body_name: base_link
      debug_vis: true
```
