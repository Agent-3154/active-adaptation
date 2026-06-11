# Repository Guidelines

## Project Structure & Module Organization
`active_adaptation/` contains the core package: environments in `envs/`, RL code in `learning/`, shared helpers in `utils/`, `sensors/`, and `project_loading/`. Hydra config lives under `cfg/` with shared defaults in `cfg/base/`, experiments in `cfg/exp/`, and task definitions in `cfg/task/`. Runtime entry points are in `scripts/` (`train_ppo.py`, `eval.py`, `play.py`, `launch_ddp.sh`). Extension projects live in `projects/` and register through `pyproject.toml`. Large robot and scene assets are expected under `.cache/aa-robot-models/`, not committed into the package.

## Build, Test, and Development Commands
Install in a Python 3.11 environment with `pip install -e .`.
Use `aa-discover-projects` to refresh discovered project/task metadata, and `aa-list-tasks` to inspect available task IDs.
Typical workflows:

```bash
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo
python scripts/eval.py task=Go2/Go2Flat algo=ppo eval_render=true
python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
bash scripts/launch_ddp.sh 0,1 train_ppo.py task=G1/G1LocoFlat algo=ppo
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for modules/functions, PascalCase for classes, and concise docstrings only where behavior is not obvious. Keep Hydra config keys and task names consistent with existing patterns such as `Go2/Go2Flat` and `ppo_symaug`. There is no pinned formatter in this repo today; keep imports grouped cleanly and match surrounding file structure. `pyproject.toml` enables Pyright checks, so prefer type-safe changes and preserve annotated APIs.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with focused runnable checks:

```bash
python scripts/train_ppo.py task=Go2/Go2Flat algo=ppo task.num_envs=16 wandb.mode=disabled
python scripts/play.py task=Go2/Go2Flat algo=ppo checkpoint_path=/path/to/checkpoint.pt
pyright active_adaptation
```

For new features, add the smallest reproducible command in the PR description and verify both config loading and the affected training/eval path.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `cleanup rewards` or `fix ground height query`. Keep commit subjects brief, lowercase, and specific to one change. PRs should explain the motivation, list the main files touched, include exact validation commands, and attach screenshots or videos for behavior/visualization changes. Link the relevant issue, experiment run, or WandB run when applicable.

---

## MDP API (V1 → V2)

The environment builds MDP components from task YAML (`cfg/task/...`). Each component is registered via `RegistryMixin` and instantiated with `ClassName.make(_target_, ...)`.

### V1 (legacy, env-bound at construction)

| Component | Base class | Constructor | Env binding |
|-----------|------------|-------------|-------------|
| Command | `Command` | `__init__(env, ...)` | immediate |
| Reward | `Reward` | `__init__(env, weight, ...)` | immediate (deprecated; raises) |
| Action | `Action` | `__init__(env)` | immediate |
| Observation | `Observation` | `__init__(env)` | immediate |

### V2 (deferred initialization)

| Component | Base class | Constructor | Env binding |
|-----------|------------|-------------|-------------|
| Command | `CommandV2` | `__init__(...)` only | `_initialize(env)` at env startup |
| Reward | `RewardV2` | `__init__(weight, ...)` only | `_initialize(env)` at env startup |
| Action | `ActionV2` | `__init__(...)` only | `_initialize(env)` at env startup |
| Observation | `ObservationV2` | `__init__(...)` only | `_initialize(env)` at env startup |
| Randomization | `RandomizationV2` | `__init__(...)` only | `_initialize(env)` at env startup |

**Why V2 exists:** components can be constructed without a simulator. This supports offline **command relabeling** (`relabel_command`, `get_state`) and **reward relabeling** (`relabel`) on stored rollouts.

**Instantiation in `env_base`:** the env tries V1 `.make(cls, env, **kwargs)` first, then V2 `.make(cls, **kwargs)` + `._initialize(env)`.

**Subclass pattern:**

```python
class MyReward(RewardV2[SingleEEFLocoManip]):
    def __init__(self, weight: float, sigma: float = 0.1):
        super().__init__(weight)
        self.sigma = sigma

    def _initialize(self, env):
        super()._initialize(env)
        self.asset = self.command_manager.asset  # env-bound handles here

    def _compute(self) -> torch.Tensor:
        ...  # uses self.command_manager, self.env

    def relabel(self, tensordict) -> torch.Tensor:
        ...  # offline replay from tensordict["command_state"]
```

Reward terms implement `_compute()`; `compute()` applies `weight`, optional per-env `modifier`, and EMA logging. Command-specific rewards should type-parameterize `RewardV2[YourCommand]`.

---

## Command lifecycle: `sync_state` vs `update`

Each env step, **after physics**:

```
sync_state()  →  rewards / terminations  →  update()  →  observations
```

| Method | When | May change commands? | Typical work |
|--------|------|----------------------|--------------|
| `sync_state()` | Before reward | **No** | Read robot state; compute tracking errors, diffs, metrics from **current** targets |
| `update()` | After reward, before obs | **Yes** | Resample commands, advance world targets, phase transitions; refresh obs-facing fields |

**Rollout / relabel implication:** rewards align with post-physics state from `sync_state()`. Call `get_state()` after `sync_state()` (not before the env step) when saving `command_state` for relabeling.

`CommandV2` additionally requires abstract `sync_state()` and `update()`, plus `get_state()` / `relabel_command()` for hindsight workflows.

**Naming migration (commands):**

| Old API | New API |
|---------|---------|
| `update()` body that only refreshed tracking | `sync_state()` |
| `step()` body that resampled / advanced commands | `update()` |

---

## Hydra config: YAML tasks + dataclass entry scripts

Configs are split into two layers:

1. **Task configs** — still YAML under `cfg/task/<Robot>/<Task>.yaml` (`# @package task`). Define `command`, `reward`, `observation`, `input`, `termination`, `randomization`, sim settings, etc. Component entries use `_target_` for the registry class name:

   ```yaml
   command:
     _target_: SingleEEFLocoManip
     eef_body_name: grasp_point
   reward:
     manip:
       eef_pos_forward_tracking: {weight: 1.5}
   ```

2. **Entry-script configs** — Python `@dataclass` registered with Hydra `ConfigStore`. The script is the config root (`config_name="train"` etc.).

   Example from `scripts/train_ppo.py`:

   ```python
   DEFAULTS = [{"task": "Velocity"}, {"algo": "ppo"}, "_self_"]

   @dataclass
   class TrainConfig:
       defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
       headless: bool = True
       total_frames: int = 150_000_000
       wandb: WandbConfig = field(default_factory=WandbConfig)
       ...

   cs = ConfigStore.instance()
   cs.store(name="train", node=TrainConfig(hydra=HydraConf(...)))

   @hydra.main(config_path="../cfg", config_name="train", version_base=None)
   def main(cfg: TrainConfig):
       OmegaConf.resolve(cfg)
       ...
   ```

   Algo configs live beside the policy module and register into the `algo` group:

   ```python
   @dataclass
   class PPOConfig:
       _target_: str = "active_adaptation.learning.ppo.ppo_symaug.PPOPolicy"
       train_every: int = 32
       lr: float = 5e-4
       in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)

   cs.store("ppo_symaug", node=PPOConfig, group="algo")
   ```

   CLI overrides still work: `python scripts/train_ppo.py task=G1/G1LocoFlat algo=ppo_symaug total_frames=1e6`.

   Nested dataclasses (`WandbConfig`, `IsaacAppConfig`) hold grouped fields. Use OmegaConf interpolation in defaults, e.g. `project: str = "${oc.select:task.project,active_adaptation}"`.

Same dataclass pattern is used in `scripts/rollout.py`, `scripts/play.py`, `scripts/eval.py`, `scripts/relabel.py`, and algo modules such as `learning/ppo/ppo_symaug.py`, `learning/offpolicy/sac.py`.

---

## Policy construction: `from_env` / `from_state_dict`

Policies follow the same **deferred-binding** idea as MDP V2: `__init__` takes tensor specs and optional symmetry transforms, not a live env. Factories bridge from runtime objects when a sim is available.

### Current flow (`helpers.make_env_policy`)

```python
policy_cls = hydra.utils.get_class(cfg.algo._target_)  # e.g. PPOPolicy
policy = policy_cls.from_env(cfg.algo, env, device=cfg.device)
if "policy" in checkpoint_state_dict:
    policy.load_state_dict(checkpoint_state_dict["policy"])
```

`from_env` is the **only** construction path used today. It reads from the built env:

- `observation_spec`, `action_spec`, `reward_spec`
- symmetry transforms from `observation_funcs` and `action_manager` (when `sym_aug` / `CMD_KEY` apply)

Then calls `__init__(cfg, observation_spec, action_spec, reward_spec, device, cmd_transform=..., ...)`.

Reference: `PPOPolicy.from_env` in `learning/ppo/ppo_symaug.py`, `SAC.from_env` in `learning/offpolicy/sac.py`.

### Planned: env-free construction

| Factory | Status | Purpose |
|---------|--------|---------|
| `from_env(cfg, env, device)` | **In use** | Training, rollout, play — env already exists |
| `from_state_dict(state_dict, device)` | **Stub** (`pass` in `ppo_symaug`) | Rebuild policy from checkpoint metadata + weights without instantiating a sim |

`from_state_dict` will eventually consume a checkpoint that carries enough **structural** information (specs, transforms, `PPOConfig`) to call `__init__` directly, then `load_state_dict`. That enables inference, distillation, or batch jobs that only need the policy module.

### Pattern for new policies

```python
class MyPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: MyConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        obs_transform: SymmetryTransform | None = None,
        act_transform: SymmetryTransform | None = None,
    ):
        ...  # build modules from specs only

    @classmethod
    def from_env(cls, cfg: MyConfig, env: _EnvBase, device: str):
        ...  # extract specs + transforms from env, return cls(...)

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict, device: str):
        raise NotImplementedError  # until checkpoint schema is defined
```

- Keep `cfg.algo._target_` pointing at the policy **class** (not `from_env`).
- Do **not** take `env` in `__init__`; stage hooks like `on_stage_start(stage, env)` can still receive the env for DDP / distributed setup after construction.
- Register algo config with `ConfigStore` (`group="algo"`) beside the policy module.

---

## Migration guide

### 1. Reward term: `Reward` → `RewardV2`

```python
# Before
class my_rew(Reward):
    def __init__(self, env, weight, sigma=0.1):
        super().__init__(env, weight)
        self.sigma = sigma

    def _compute(self):
        return self.command_manager.pos_error_norm

# After
class my_rew(RewardV2[MyCommand]):
    def __init__(self, weight: float, sigma: float = 0.1):
        super().__init__(weight)
        self.sigma = sigma

    def _initialize(self, env):
        super()._initialize(env)

    def _compute(self):
        return self.command_manager.pos_error_norm

    def relabel(self, tensordict):
        return tensordict["command_state", "pos_error_norm"]
```

- Remove `env` from `__init__`; move sim/command handles to `_initialize`.
- Add `relabel()` if the term participates in offline reward relabeling.
- Task YAML unchanged (`weight`, kwargs still map to `__init__`).

### 2. Command: split tracking vs mutation

```python
# Before (everything in update, or update + step)
def update(self):
    self._read_robot_state()
    self._maybe_resample_commands()   # wrong: affects this step's reward
    self._compute_tracking_errors()

def step(self):
    self._advance_targets()

# After
def sync_state(self):
    self._read_robot_state()
    self._sync_frames_from_targets()  # no resample, no advance
    self._compute_tracking_errors()

def update(self):
    self._maybe_resample_commands()
    self._advance_targets()
    self._sync_frames_from_targets()
    self._compute_tracking_errors()
```

For `Command` (v1, e.g. `Twist`): same split — implement `sync_state()` and `update()`; remove `step()`.

For `CommandV2`: implement `get_state()` (snapshot after `sync_state`) and `relabel_command()` when used as a relabel target.

### 3. Action / observation / randomization → V2

```python
# Before
class MyAction(Action):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

# After
class MyAction(ActionV2):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def _initialize(self, env):
        super()._initialize(env)
        # allocate buffers using self.num_envs, self.device
```

### 4. Entry script config: YAML root → dataclass

When adding or porting a script config:

1. Define a `@dataclass` root config (e.g. `TrainConfig`) in the script or algo module.
2. Set `defaults` to pull in `task` and `algo` YAML groups plus `"_self_"`.
3. Register with `ConfigStore.instance().store(name="...", node=...)`.
4. Type the Hydra entrypoint as `def main(cfg: TrainConfig)`.
5. Call `OmegaConf.resolve(cfg)` and `OmegaConf.set_struct(cfg, False)` at the top of `main`.
6. Keep task-specific MDP structure in `cfg/task/*.yaml`; only move script-level knobs (frames, WandB, checkpoints, headless) into the dataclass.

### 5. Policy: direct `__init__(env, ...)` → `from_env`

```python
# Before (helpers or policy module)
policy = PPOPolicy(cfg.algo, env, device=cfg.device)

# After
policy = PPOPolicy.from_env(cfg.algo, env, device=cfg.device)
```

- Move env-derived inputs (specs, symmetry transforms) into `from_env`.
- Keep `__init__` env-free so `from_state_dict` can be implemented later.
- Checkpoint loading stays separate: `from_env` then `load_state_dict(checkpoint["policy"])`.

### 6. Checklist for new tasks / components

- [ ] Command implements `sync_state` (tracking only) and `update` (mutations only).
- [ ] Rewards use `RewardV2` with `_initialize` + `relabel` when rollouts are relabeled.
- [ ] `CommandV2` provides `get_state()` if used with `scripts/relabel.py` or rollout archives.
- [ ] `_target_` in task YAML matches the registered class name.
- [ ] `supported_backends` on the component matches your target backend.
- [ ] Entry script has a registered dataclass config if it uses `@hydra.main`.
- [ ] Policy class exposes `from_env`; `__init__` does not require an env instance.

### 7. Common pitfalls

- **Putting resampling in `sync_state`** — rewards see commands that were not active during the physics step.
- **Calling `update()` in `_initialize`** — use `sync_state()` for initial tracking buffers; reserve `update()` for post-reward command setup.
- **Mixing V1/V2 constructors** — V2 `.make()` does not take `env`; the env calls `_initialize` separately.
- **Forgetting `relabel()`** — `scripts/relabel.py` skips reward groups already present; new relabel groups need working `relabel()` implementations.
- **Pre-step `get_state()` in rollout** — capture after `sync_state` (post-physics), not before `env.step()`.
- **Passing `env` into policy `__init__`** — use `from_env` so specs/transforms are explicit and `from_state_dict` remains viable.
