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
