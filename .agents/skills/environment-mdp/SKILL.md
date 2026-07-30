---
name: environment-mdp
description: Implement and wire MDP terms in active-adaptation (observations, rewards, terminations, actions, commands, randomizations) using the V2 deferred-init API. Use when adding or modifying classes under envs/mdp/, editing cfg/task/ observation/reward/termination/input/command/randomization blocks, debugging env_base step/reset callbacks, cross-backend debug viz (scene.draw_*, camera frustums), Warp/simple-raycaster raycasting, USD mesh extraction, or Isaac/mjlab Viser viewers.
---

# Environment / MDP terms (active-adaptation)

Implement MDP terms as **V2** classes: construct from Hydra kwargs **without** an env, then bind in `_initialize(env)` after the scene exists. Register via `RegistryMixin`; task YAML selects them by class name.

**Primary references**
- Env wiring + step/reset: `active_adaptation/envs/env_base.py`
- Shared hooks: `active_adaptation/envs/mdp/base.py` (`MDPComponent`)
- Term bases: `envs/mdp/{observations,rewards,terminations,actions,commands,randomizations}/base.py`
- Task configs: `cfg/task/**/*.yaml`
- Debug viz API: `active_adaptation/envs/adapters.py` (`SceneAdapter`, `CameraFrustumHandle`)
- Raycasting / USD meshes: installed package `simple-raycaster` (see section below + [reference.md](reference.md))

Read [reference.md](reference.md) for the step-loop diagram, callback registration, file map, viz backends, and raycast/mesh details.

**Related skills:** `onpolicy-algorithms`, `offpolicy-algorithms`.

---

## When to use

- Adding a new observation / reward / termination / action / command / randomization
- Porting a legacy `Reward`/`Observation`/… term to V2
- Wiring or renaming terms in `cfg/task/`
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

---

## Checklist: new term

```
Task Progress:
- [ ] Choose term type and V2 base class
- [ ] Implement class in the matching envs/mdp/<family>/ module
- [ ] __init__: config kwargs only (no env / asset access)
- [ ] _initialize: super()._initialize(env); then asset/sensor/buffer setup
- [ ] Implement the type-specific compute / apply / sync API
- [ ] Optional lifecycle: update, reset(env_ids, tensordict), pre_step, post_step, startup, debug_draw
- [ ] If `debug_draw`: use `env.scene.draw_*` / `create_camera_frustum` (not `env.debug_draw`)
- [ ] Optional: symmetry_transform (obs/action) for symaug
- [ ] Ensure module is imported (auto-import or explicit in package __init__)
- [ ] Wire into cfg/task/ YAML
- [ ] Smoke: instantiate env and step once (or train a few iters)
```

---

## Term APIs (what to implement)

| Type | Config block | Factory | Must implement | Notes |
|------|--------------|---------|----------------|-------|
| Observation | `observation.<group>.<name>` | `ObservationV2.make` | `compute() -> Tensor` | Groups concat along last dim in YAML order |
| Reward | `reward.<group>.<name>` | `RewardV2.make` | `_compute() -> Tensor` or `(rew, is_active)` | `compute()` applies `weight` + modifier + EMA |
| Termination | `termination.<name>` | `TerminationV2.make` | `compute(terminated) -> bool Tensor` or `(term, discount)` | `is_timeout=True` → truncated |
| Action | `input.<name>` | `ActionV2.make` | `process_action`, `apply_action` | Set `action_dim`; often `symmetry_transform` |
| Command | `command` | `CommandV2.make` | `sync_state`, `update` | Also `sample_init` for resets; see step order |
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

### USD → trimesh (body-local visuals)

Extraction lives in `simple_raycaster.utils_usd` (also used by Viser mesh upload):

1. `find_matching_prims(regex, stage)` — stage traverse with anchored regex.
2. `get_trimesh_from_prim(prim)` — collect `Mesh`/`Cube` under the prim (follows instance prototypes), convert via `usd2trimesh`, apply **local** transform relative to the parent prim (`world * parent⁻¹`), concatenate + `merge_vertices`.
3. Result is in **body / parent frame**; at runtime multiply by `body_link_pose_w` (or let V2 do it).

Static terrain: combine under e.g. `/World/ground` and keep identity pose (geometry already world-framed).

MuJoCo counterpart: `utils_mjc.get_trimesh_from_body` + `MultiMeshRaycaster.from_MjModel`.

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
- Forgetting to import the module (class never enters the registry)
- Returning unbatched reward/obs tensors
- Registering empty `update`/`reset` overrides unintentionally (they still run every step/reset)
- Using `env.debug_draw` instead of `env.scene.draw_*` / `create_camera_frustum`
