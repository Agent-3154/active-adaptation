# Environment / MDP — reference

## File map

```
active_adaptation/envs/
├── env_base.py              # _EnvBase: create / init / step / reset wiring
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
    │   ├── common.py, joint.py, …
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
  ├─ command.sync_state()
  ├─ update callbacks
  ├─ _compute_reward
  ├─ _compute_termination
  ├─ command.update()
  ├─ _compute_observation
  └─ debug_draw callbacks (if GUI)
```

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

### Viser (Isaac) — reuse the same meshes

When building an Isaac Viser viewer (mjlab-parity: robot mesh + camera frustums):

1. Extract body visuals the same way as `MultiMeshRaycasterV2.add_isaac_entity` / `get_trimesh_from_prim`.
2. Upload once (`add_batched_meshes_*` or equivalent); do not re-parse USD every step.
3. Each viewer update: write `entity.data.body_link_pose_w` (pos + WXYZ→Viser wxyz) for all envs or the selected env.
4. Camera obs debug: attach images to Viser **camera frustum** helpers (pose from sensor/body offset), not only flat GUI image panels.
5. Keep Omniverse Kit viewport optional; Viser is the headless/browser path analogous to `MjLabViewer`.
