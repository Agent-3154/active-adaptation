---
name: environment-mdp
description: Implement and wire MDP terms in active-adaptation (observations, rewards, terminations, actions, commands, randomizations) using the V2 deferred-init API. Use when adding or modifying classes under envs/mdp/, editing cfg/task/ observation/reward/termination/input/command/randomization blocks, or debugging env_base step/reset callbacks.
---

# Environment / MDP terms (active-adaptation)

Implement MDP terms as **V2** classes: construct from Hydra kwargs **without** an env, then bind in `_initialize(env)` after the scene exists. Register via `RegistryMixin`; task YAML selects them by class name.

**Primary references**
- Env wiring + step/reset: `active_adaptation/envs/env_base.py`
- Shared hooks: `active_adaptation/envs/mdp/base.py` (`MDPComponent`)
- Term bases: `envs/mdp/{observations,rewards,terminations,actions,commands,randomizations}/base.py`
- Task configs: `cfg/task/**/*.yaml`

Read [reference.md](reference.md) for the step-loop diagram, callback registration, and file map.

**Related skills:** `onpolicy-algorithms`, `offpolicy-algorithms`.

---

## When to use

- Adding a new observation / reward / termination / action / command / randomization
- Porting a legacy `Reward`/`Observation`/… term to V2
- Wiring or renaming terms in `cfg/task/`
- Debugging missing callbacks, wrong step order, or registry lookup failures

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

1. `command_manager.sync_state()` — refresh intermediates; **do not** change commands
2. `update` callbacks (obs/reward/rand/… that override `update`)
3. rewards → terminations
4. `command_manager.update()` — may resample / change commands
5. observations

Inside each physics substep: `process_action` (once) → `apply_action` → `pre_step` → sim → `post_step`.

First `_reset`: run `startup` callbacks once, then for reset envs: `_reset_idx` → `scene.reset` → `reset` callbacks.

### Command timing

Rewards/terminations that depend on command-derived state must read values produced in `sync_state` (or earlier). Observations may see post-`update` commands. Do not put reward-critical refreshes only in `update`.

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

## Anti-patterns

- Accessing `env.scene` / assets in `__init__` (scene does not exist yet)
- Subclassing legacy `Reward` / `Observation` / …
- Putting command resampling in `sync_state`
- Forgetting to import the module (class never enters the registry)
- Returning unbatched reward/obs tensors
- Registering empty `update`/`reset` overrides unintentionally (they still run every step/reset)
