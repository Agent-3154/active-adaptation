from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from typing import Callable, Dict, Mapping, Any, Optional, cast
from dataclasses import dataclass

import numpy as np
import torch
import time
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.envs.adapters import SimAdapter, SceneAdapter
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.video_recorder import (
    VideoRecorder,
    NullVideoRecorder,
    IsaacVideoRecorder,
    RgbArrayVideoRecorder,
)
from active_adaptation.envs.utils import GroundQuery
from active_adaptation.registry import RegistryMixin

if active_adaptation.get_backend() == "isaaclab":
    import isaacsim.core.utils.torch as torch_utils


EMA_DECAY = 0.99
PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _sanitize_nonfinite_env_rows(tensordict: TensorDictBase) -> torch.Tensor:
    """Zero invalid transition rows and terminate those environments."""
    num_envs = tensordict.batch_size[0]
    invalid = torch.zeros(num_envs, dtype=torch.bool, device=tensordict.device)
    floating_leaves = []
    for key, value in tensordict.items(include_nested=True, leaves_only=True):
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.ndim > 0
            and value.shape[0] == num_envs
        ):
            floating_leaves.append((key, value))
            invalid |= ~torch.isfinite(value).reshape(num_envs, -1).all(dim=1)

    if not invalid.any():
        return invalid

    for key, value in floating_leaves:
        row_mask = invalid.reshape(num_envs, *((1,) * (value.ndim - 1)))
        tensordict.set(key, torch.where(row_mask, torch.zeros_like(value), value))
    tensordict["terminated"][invalid] = True
    tensordict["truncated"][invalid] = False
    tensordict["done"][invalid] = True
    tensordict["discount"][invalid] = 0.0
    return invalid


def parse_component_spec(name: str, cfg):
    if cfg is None or not hasattr(cfg, "items"):
        raise ValueError(f"Component '{name}' must be a mapping.")
    kwargs = dict(cfg)
    target = kwargs.pop("_target_", name)
    return name, target, kwargs


def _tensordict_to_composite(td: TensorDictBase) -> Composite:
    """Build a :class:`Composite` spec matching a compact observation TensorDict."""
    spec: dict = {}
    for key, value in td.items():
        if isinstance(value, TensorDictBase):
            spec[key] = _tensordict_to_composite(value)
        else:
            spec[key] = Unbounded(value.shape, dtype=value.dtype, device=td.device)
    return Composite(spec, shape=td.batch_size, device=td.device)


class ObsGroup:
    def __init__(
        self,
        name: str,
        funcs: Dict[str, mdp.Observation],
        max_delay: int = 0,
    ):
        self.name = name
        self.funcs = funcs
        self.max_delay = max_delay
        self.timestamp = -1
        self._keys = list(funcs.keys())
        self._is_functional: bool | None = None

    def _initialize(self, env: "_EnvBase"):
        self.env = env
        for func in self.funcs.values():
            func._initialize(env)

        flags = [bool(func.functional) for func in self.funcs.values()]
        if any(flags) and not all(flags):
            raise ValueError(
                f"ObsGroup '{self.name}' mixes functional and dense terms; "
                "use all-functional or all-dense within a group."
            )
        self._is_functional = bool(flags) and bool(flags[0])

        shapes = OrderedDict()
        if self._is_functional:
            compact = OrderedDict()
            for key, func in self.funcs.items():
                term_td = TensorDict(
                    {}, batch_size=[env.num_envs], device=self.env.device
                )
                func.fupdate(term_td)
                compact[key] = term_td
                shapes[key] = func.fcompute(term_td).shape
            group_td = TensorDict(
                compact, batch_size=[env.num_envs], device=self.env.device
            )
            spec = {self.name: _tensordict_to_composite(group_td)}
        else:
            outputs = OrderedDict(
                (key, func.compute()) for key, func in self.funcs.items()
            )
            for key, tensor in outputs.items():
                shapes[key] = tensor.shape
            obs = torch.cat(list(outputs.values()), dim=-1)
            spec = {self.name: Unbounded(obs.shape, dtype=obs.dtype)}

        self._spec = Composite(spec, shape=[env.num_envs]).to(self.env.device)
        self._shapes = shapes

    def __getitem__(self, key: str) -> mdp.Observation:
        return self.funcs[key]

    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        return self._spec

    @property
    def shapes(self):
        return self._shapes

    @property
    def split(self):
        return [shape[-1] for shape in self._shapes.values()]

    @property
    def is_functional(self) -> bool:
        if self._is_functional is None:
            raise RuntimeError(f"ObsGroup '{self.name}' is not initialized")
        return self._is_functional

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> TensorDictBase:
        if self._is_functional:
            compact = OrderedDict()
            for key, func in self.funcs.items():
                term_td = TensorDict(
                    {},
                    batch_size=tensordict.batch_size,
                    device=tensordict.device,
                )
                func.fupdate(term_td)
                compact[key] = term_td
            tensordict[self.name] = TensorDict(
                compact,
                batch_size=tensordict.batch_size,
                device=tensordict.device,
            )
        else:
            outputs = OrderedDict(
                (key, func.compute()) for key, func in self.funcs.items()
            )
            tensordict[self.name] = torch.cat(list(outputs.values()), dim=-1)
        return tensordict

    def materialize(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Densify a functional group stored under ``self.name`` into a cat vector.

        Idempotent: if the group entry is already a dense tensor, return it.

        ``nan_to_num`` covers the post-reset zeroed compact buffer (first obs is
        discarded by algos); zero quats would otherwise NaN in ``fcompute``.
        """
        value = tensordict[self.name]
        if not self._is_functional or torch.is_tensor(value):
            return value
        parts = [
            func.fcompute(value[key]) for key, func in self.funcs.items()
        ]
        return torch.nan_to_num(torch.cat(parts, dim=-1))

    def symmetry_transform(self):
        """Return the mirror transform for the concatenated observation group.

        Each observation component defines a local
        :class:`~active_adaptation.utils.symmetry.SymmetryTransform` matching
        the tensor slice produced by that component's ``compute()`` /
        densified ``fcompute()`` output. ``ObsGroup`` concatenates observations
        in ``self.funcs`` order, so the full group transform is the
        concatenation of the same per-component transforms in the same order.

        This is used by symmetry augmentation/equivariance losses to mirror a
        complete policy observation without each learner needing to know how the
        observation was assembled. When adding a new observation term, implement
        its ``symmetry_transform()`` with the same dimension, permutation, and
        sign convention as its densified output tensor.

        For functional groups the transform applies to the densified vector from
        :meth:`materialize` (same layout as a dense group).
        """
        transforms = [
            func.symmetry_transform().to(func.device) for func in self.funcs.values()
        ]
        return symmetry_utils.SymmetryTransform.cat(transforms)


class RewardGroup:
    """Group of reward terms; per-term EMA logging lives on each :class:`~mdp.Reward`."""

    def __init__(
        self,
        name: str,
        funcs: OrderedDict[str, mdp.Reward],
        enabled: bool = True,
        compile: bool = False,
    ):
        self.name = name
        self.funcs = funcs
        self.enabled = enabled
        self.compile = compile
        self.enabled_rewards = sum(func.enabled for func in funcs.values())
    
    def _initialize(self, env: "_EnvBase"):
        self.env = env
        for func in self.funcs.values():
            func._initialize(env)
        if self.compile:
            self.compute = torch.compile(self.compute, fullgraph=True)
    
    def __getitem__(self, key: str) -> mdp.Reward:
        return self.funcs[key]

    def compute(self) -> torch.Tensor:
        rewards = []
        for key, func in self.funcs.items():
            with ScopedTimer(f"{self.name}.{key}", sync=False):
                reward = func.compute()
            self.env.stats[self.name, key].add_(reward)
            if func.enabled:
                rewards.append(reward)
        if len(rewards):
            return torch.cat(rewards, 1).sum(dim=1, keepdim=True)
        return torch.zeros(self.env.num_envs, 1, device=self.env.device)

    def get_ema_stats(self) -> Dict[str, float]:
        """Flatten per-term EMA metrics (e.g. mean, optional var) for logging."""
        result: Dict[str, float] = {}
        for key, func in self.funcs.items():
            mean, var = func.get_ema_stats()
            result[key] = mean.item()
            if var is not None:
                result[f"{key}_var"] = var.item()
        return result
    
    def relabel(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Relabel the reward group."""
        T, N = tensordict.shape[:2]
        rew = torch.zeros(T, N, 1, device=tensordict.device)
        for name, func in self.funcs.items():
            rew = rew + func.weight * func.relabel(tensordict)
        return rew.reshape(T, N, 1)
    
    @classmethod
    def create_from(
        cls,
        group_name: str,
        group_cfg: dict,
        *,
        make_component: Callable[[type[mdp.Reward], str, dict], mdp.Reward | None],
        register_component: Callable[[mdp.MDPComponent], None] | None = None,
    ) -> "RewardGroup":
        print(f"Reward group: {group_name}")
        funcs: OrderedDict[str, mdp.Reward] = OrderedDict()

        group_cfg = dict(group_cfg)
        enabled = group_cfg.pop("_enabled_", True)
        compile = group_cfg.pop("_compile_", False)

        for rew_name, rew_cfg in group_cfg.items():
            rew_name, cls_name, rew_kwargs = parse_component_spec(rew_name, rew_cfg)
            reward = make_component(mdp.Reward, cls_name, rew_kwargs)
            if not reward:
                continue
            funcs[rew_name] = reward
            if register_component is not None and isinstance(reward, mdp.MDPComponent):
                register_component(reward)
            print(f"\t{rew_name}: \t{reward.weight:.2f}")

        return cls(group_name, funcs, enabled, compile)

@dataclass
class EnvConfig:
    # common terms
    num_envs: int
    max_episode_length: int
    
    # simulation terms, maybe backend-specific
    sim: Any
    robot: Any
    terrain: Any
    sensors: Optional[Dict[str, Any]]
    objects: Optional[Dict[str, Any]]


class _EnvBase(EnvBase, RegistryMixin):
    def __init__(self, cfg: EnvConfig, device: str, headless: bool = True):
        super().__init__(
            device=device,
            batch_size=[cfg.num_envs],
            run_type_checks=False,
        )
        self.backend = active_adaptation.get_backend()
        self.cfg = cfg
        self.headless = headless

        self._create_mdp_terms()

        self.terrain_type = None
        self.visual = None
        self.setup_scene()
        self.sim = cast(SimAdapter, self.sim)
        self.scene = cast(SceneAdapter, self.scene)
        self._setup_visual()
        if self.terrain_type is None:
            warnings.warn(
                "Terrain type is not set. Please check if the scene is properly initialized."
            )
        self.step_dt = float(self.cfg.sim.step_dt)
        self.physics_dt = float(self.sim.get_physics_dt())
        self.decimation = int(self.step_dt / self.physics_dt)

        self.max_episode_length = int(self.cfg.max_episode_length)

        with torch.device(self.device):
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
            self.episode_id = torch.zeros(self.num_envs, dtype=torch.long)
            self.episode_origin = torch.zeros(self.num_envs, 3)

        self.episode_count = 0
        self.current_iter = 0

        self._sensor_render_enabled = False
        self._last_gui_render_time = 0.0
        self._initialize_mdp_terms()
        self._build_tensor_specs()

        self.timestamp: int = 0
        self.stats: TensorDict = self.reward_spec["stats"].zero()
        self.input_tensordict = None
        self.extra = {}
        self._nonfinite_rows_total = 0
        [callback() for callback in self._startup_callbacks]
        self._startup_done = True

    @property
    def max_episode_length(self) -> torch.Tensor:
        return self._max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, value: int | torch.Tensor):
        if isinstance(value, int):
            value = torch.full((self.num_envs, 1), value, device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.dtype == torch.long, "Max episode length must be an integer tensor"
            assert value.shape == (self.num_envs, 1), "Max episode length must be a tensor of shape (num_envs, 1)"
        else:
            raise ValueError(f"Invalid type for max episode length: {type(value)}")
        self._max_episode_length = value.to(self.device)

    @property
    def sensor_render_enabled(self) -> bool:
        """When True, call :meth:`SimAdapter.render_sensors` each control step.

        Native Kit / mjlab camera observations set this in ``_initialize``.
        3DGS cameras (:class:`~active_adaptation.envs.mdp.observations.visual.gs_camera`)
        call ``env.visual.render`` in ``compute`` instead (option A) and do **not**
        need this flag.
        """
        return self._sensor_render_enabled

    @sensor_render_enabled.setter
    def sensor_render_enabled(self, value: bool):
        self._sensor_render_enabled = bool(value)

    @property
    def render_enabled(self) -> bool:
        """Deprecated alias for :attr:`sensor_render_enabled`."""
        return self.sensor_render_enabled

    @render_enabled.setter
    def render_enabled(self, value: bool):
        self.sensor_render_enabled = value

    def _setup_visual(self) -> None:
        """Optional photoreal world (3DGS + mesh composite). Collision on ``scene``."""
        visual_cfg = self.cfg.get("visual", None)
        if visual_cfg is None:
            self.visual = None
            return
        from active_adaptation.envs.visual import make_visual_world

        self.visual = make_visual_world(visual_cfg, device=self.device)
        if self.visual is None:
            return
        # Eager load so missing PLY fails at env construction, not first obs.
        self.visual.load()
        # Body-local visuals for GS mesh composite (typically ``robot``).
        attach = getattr(self.visual, "attach_scene_meshes", None)
        if attach is not None:
            attach(self.scene)

    def _create_mdp_terms(self):
        self._scene_components: list[mdp.MDPComponent] = []
        self.randomizations: Mapping[str, mdp.Randomization] = OrderedDict()
        self.observation_groups: Mapping[str, ObsGroup] = OrderedDict()
        self.reward_groups: Mapping[str, RewardGroup] = OrderedDict()
        self.input_managers: Mapping[str, mdp.Action] = OrderedDict()
        self.termination_funcs: Mapping[str, mdp.Termination] = OrderedDict()

        self._enabled_reward_groups = 0

        self._startup_callbacks = []
        self._reset_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []
        self._update_callbacks = []
        self._debug_draw_callbacks = []

        # MDP: command manager
        command_cfg = dict(self.cfg.command)
        class_name = command_cfg.pop("_target_", None)
        command = self._make_component(mdp.Command, class_name, command_cfg)
        if not command:
            raise ValueError(f"Command class '{class_name}' not found")
        self.command_manager = command
        if isinstance(command, mdp.MDPComponent):
            self._scene_components.append(command)

        # MDP: input managers
        for input_name, input_cfg in dict(self.cfg.get("input", {})).items():
            _, input_cls_name, input_kwargs = parse_component_spec(
                input_name, input_cfg
            )
            input_manager = self._make_component(
                mdp.Action, input_cls_name, input_kwargs
            )
            if not input_manager:
                continue
            self.input_managers[input_name] = input_manager
            if isinstance(input_manager, mdp.MDPComponent):
                self._scene_components.append(input_manager)

        # MDP: randomizations
        for rand_name, rand_cfg in self.cfg.get("randomization", {}).items():
            rand_name, cls_name, rand_kwargs = parse_component_spec(rand_name, rand_cfg)
            rand = self._make_component(mdp.Randomization, cls_name, rand_kwargs)
            if not rand:
                continue
            self.randomizations[rand_name] = rand
            if isinstance(rand, mdp.MDPComponent):
                self._add_mdp_component(rand)

        # MDP: observations
        for group_name, group_cfg in self.cfg.observation.items():
            funcs = OrderedDict()
            for obs_name, obs_cfg in group_cfg.items():
                obs_name, obs_cls_name, obs_kwargs = parse_component_spec(
                    obs_name, obs_cfg
                )
                obs = self._make_component(mdp.Observation, obs_cls_name, obs_kwargs)
                if not obs:
                    continue
                funcs[obs_name] = obs
                if isinstance(obs, mdp.MDPComponent):
                    self._add_mdp_component(obs)
            self.observation_groups[group_name] = ObsGroup(group_name, funcs)

        # MDP: rewards
        reward_cfg = dict(self.cfg.reward)
        self.mult_dt = reward_cfg.pop("_mult_dt_", True)
        for group_name, group_cfg in reward_cfg.items():
            rg = RewardGroup.create_from(
                group_name,
                group_cfg,
                make_component=self._make_component,
                register_component=self._add_mdp_component,
            )
            self._enabled_reward_groups += int(rg.enabled)
            self.reward_groups[group_name] = rg

        # MDP: terminations
        termination_cfg = dict(self.cfg.get("termination", {}))
        for term_name, term_cfg in termination_cfg.items():
            term_name, cls_name, term_kwargs = parse_component_spec(term_name, term_cfg)
            term = self._make_component(mdp.Termination, cls_name, term_kwargs)
            if not term:
                continue
            self.termination_funcs[term_name] = term
            if isinstance(term, mdp.MDPComponent):
                self._add_mdp_component(term)

    def _edit_scene_spec(self, scene_cfg: Any) -> None:
        for component in self._scene_components:
            component.edit_spec(scene_cfg)

    def _initialize_mdp_terms(self):
        self.command_manager._initialize(self)
        self._register_command_component(self.command_manager)
        for input_manager in self.input_managers.values():
            input_manager._initialize(self)
            self._add_mdp_component(input_manager)
        for rand in self.randomizations.values():
            rand._initialize(self)
            self._add_mdp_component(rand)
        for group in self.observation_groups.values():
            group._initialize(self)
            for obs in group.funcs.values():
                self._add_mdp_component(obs)
        for reward_group in self.reward_groups.values():
            reward_group._initialize(self)
            for reward in reward_group.funcs.values():
                self._add_mdp_component(reward)
        for term in self.termination_funcs.values():
            term._initialize(self)
            self._add_mdp_component(term)

    def _make_component(
        self,
        base_cls: type[mdp.MDPComponent],
        class_name: str,
        kwargs: dict[str, Any],
    ) -> mdp.MDPComponent | None:
        if class_name not in base_cls.registry:
            raise ValueError(f"Class '{class_name}' not found in {base_cls.__name__}.registry")
        instance_cls = base_cls.registry[class_name]
        backend = active_adaptation.get_backend()
        if backend is not None and backend not in instance_cls.supported_backends:
            warnings.warn(
                f"Class '{class_name}' does not support backend '{backend}'. "
                f"Supported backends: {instance_cls.supported_backends}"
            )
            return None
        return instance_cls(**kwargs)

    def _register_command_component(self, command: mdp.Command) -> None:
        if mdp.is_method_implemented(command, mdp.MDPComponent, "startup"):
            self._startup_callbacks.append(command.startup)
        if mdp.is_method_implemented(command, mdp.MDPComponent, "reset"):
            self._reset_callbacks.append(command.reset)
        if mdp.is_method_implemented(command, mdp.MDPComponent, "pre_step"):
            self._pre_step_callbacks.append(command.pre_step)
        if mdp.is_method_implemented(command, mdp.MDPComponent, "post_step"):
            self._post_step_callbacks.append(command.post_step)
        if mdp.is_method_implemented(command, mdp.MDPComponent, "debug_draw"):
            self._debug_draw_callbacks.append(command.debug_draw)

    def _build_tensor_specs(self):
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            shape=[self.num_envs],
            device=self.device,
        )

        action_spec = {
            input_name: Unbounded(
                [self.num_envs, input_manager.action_dim], device=self.device
            )
            for input_name, input_manager in self.input_managers.items()
        }
        self.action_spec = Composite(
            action_spec, shape=[self.num_envs], device=self.device
        )

        observation_spec = {}
        [
            observation_spec.update(group.spec)
            for group in self.observation_groups.values()
        ]
        self.observation_spec = Composite(
            observation_spec, shape=[self.num_envs], device=self.device
        )
        self.observation_spec["episode_id"] = Unbounded(
            [self.num_envs], dtype=torch.long, device=self.device
        )

        reward_spec = Composite({})

        scalar = Unbounded(1, device=self.device)
        for group_name, reward_group in self.reward_groups.items():
            if reward_group.enabled:
                reward_spec["reward", group_name] = scalar.clone()
            for rew_name in reward_group.funcs.keys():
                reward_spec["stats", group_name, rew_name] = scalar.clone()
            reward_spec["stats", group_name, "return"] = scalar.clone()

        for term_name in self.termination_funcs.keys():
            reward_spec["stats", "termination", term_name] = scalar.clone()

        reward_spec["discount"] = Unbounded(1, device=self.device)
        reward_spec["stats", "success"] = scalar.clone()
        reward_spec["stats", "episode_len"] = scalar.clone()
        self.reward_spec = reward_spec.expand(self.num_envs).to(self.device)

    def _add_mdp_component(self, component: mdp.MDPComponent):
        if component not in self._scene_components:
            self._scene_components.append(component)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "startup"):
            self._startup_callbacks.append(component.startup)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "reset"):
            self._reset_callbacks.append(component.reset)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "pre_step"):
            self._pre_step_callbacks.append(component.pre_step)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "post_step"):
            self._post_step_callbacks.append(component.post_step)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "update"):
            cb = ScopedTimer(component.__class__.__name__)(component.update)
            self._update_callbacks.append(cb)
        if mdp.is_method_implemented(component, mdp.MDPComponent, "debug_draw"):
            self._debug_draw_callbacks.append(component.debug_draw)

    def setup_scene(self):
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Runtime helpers
    # ---------------------------------------------------------------------
    def set_progress(self, progress: int):
        self.current_iter = progress

    @staticmethod
    def _callback_label(callback) -> str:
        owner = getattr(callback, "__self__", None)
        if owner is not None:
            return owner.__class__.__name__
        return getattr(callback, "__qualname__", getattr(callback, "__name__", "callback"))

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def action_manager(self):
        return self.input_managers["action"]

    @property
    def stats_ema(self) -> Dict[str, float]:
        """Aggregate EMA stats from all reward groups."""
        result = {}
        for group_key, group in self.reward_groups.items():
            for rew_key, value in group.get_ema_stats().items():
                result[f"reward.{group_key}/{rew_key}"] = value
        return result

    @ScopedTimer("env._reset", sync=PROFILE_SYNC_TIMERS)
    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
            tensordict = TensorDict(
                {}, batch_size=[self.num_envs], device=self.device
            )

        if len(env_ids):
            num_envs = env_ids.numel()
            self.episode_length_buf[env_ids] = 0
            self.episode_id[env_ids] = self.episode_count + torch.arange(
                num_envs, device=self.device
            )
            self.episode_count += num_envs

            self._reset_idx(env_ids, tensordict)
            self.scene.reset(env_ids)
            # MDP terms: reset(env_ids, tensordict) — may read/write tensordict
            [callback(env_ids, tensordict) for callback in self._reset_callbacks]

        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        tensordict.set("episode_id", self.episode_id.clone())
        self._last_gui_render_time = time.perf_counter()
        return tensordict

    def _reset_idx(self, env_ids: torch.Tensor, reset_td: TensorDictBase):
        init_state = self.command_manager.sample_init(env_ids, reset_td)
        # ponytail: keep legacy sample_init return support until remaining AA
        # commands are migrated to write simulator state in-place.
        if init_state is None:
            self.stats[env_ids] = 0.0
            return
        if not isinstance(init_state, dict):
            init_state = {"robot": init_state}
        for key, value in init_state.items():
            entity = self.scene[key]
            if self.backend == "mjlab" and entity.is_fixed_base:
                entity.write_mocap_pose_to_sim(value[:, :7], env_ids=env_ids)
            else:
                entity.write_root_state_to_sim(value, env_ids=env_ids)
        self.stats[env_ids] = 0.0

    # TODO: add explanation for the difference
    def _should_render_sensors(self) -> bool:
        return self.sensor_render_enabled

    def _should_render_gui(self) -> bool:
        if not self.sim.has_gui():
            return False
        if time.perf_counter() - self._last_gui_render_time > 1.0 / 30.0:
            self._last_gui_render_time = time.perf_counter()
            return True
        return False

    def _should_debug_draw(self) -> bool:
        return self.sim.has_gui()

    @ScopedTimer("env._step", sync=PROFILE_SYNC_TIMERS)
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        with ScopedTimer("process_action", sync=False):
            self.command_manager.prescribe(tensordict)
            for input_key, input_manager in self.input_managers.items():
                input_manager.process_action(tensordict.get(input_key, None))
        
        with ScopedTimer("simulation", sync=False):
            for substep in range(self.decimation):
                with ScopedTimer("pre_step_callbacks", sync=False):
                    self.scene.zero_external_wrenches()
                    self._apply_action(substep)
                    [callback(substep) for callback in self._pre_step_callbacks]
                    self.scene.write_data_to_sim()
                with ScopedTimer("sim.step", sync=PROFILE_SYNC_TIMERS):
                    self.sim.step()
                    if substep == self.decimation - 1:
                        if self._should_render_sensors():
                            self.sim.render_sensors()
                        if self._should_render_gui():
                            self.sim.render_gui()
                with ScopedTimer("scene.update", sync=PROFILE_SYNC_TIMERS):
                    self.scene.update(self.physics_dt)
                    if hasattr(self.scene, "update_warp_sensors"):
                        self.scene.update_warp_sensors()
                with ScopedTimer("post_step_callbacks", sync=False):
                    [callback(substep) for callback in self._post_step_callbacks]

        self.episode_length_buf.add_(1)
        self.timestamp += 1

        tensordict = TensorDict({}, self.num_envs, device=self.device)

        with ScopedTimer("command.update", sync=False):
            self.command_manager.update()
        with ScopedTimer("update_callbacks", sync=False):
            [callback() for callback in self._update_callbacks]

        tensordict = self._compute_reward(tensordict)
        tensordict = self._compute_termination(tensordict)
        with ScopedTimer("command.step", sync=False):
            self.command_manager.step()
    
        tensordict = self._compute_observation(tensordict)

        tensordict.set("episode_id", self.episode_id.clone())
        tensordict["stats"] = self.stats.clone()

        invalid = _sanitize_nonfinite_env_rows(tensordict)
        invalid_count = int(invalid.sum().item())
        self._nonfinite_rows_total += invalid_count
        self.extra["env/nonfinite_rows"] = invalid_count
        self.extra["env/nonfinite_rows_total"] = self._nonfinite_rows_total

        if self._should_debug_draw():
            [callback() for callback in self._debug_draw_callbacks]

        return tensordict

    def _apply_action(self, substep: int):
        [
            input_manager.apply_action(substep)
            for input_manager in self.input_managers.values()
        ]

    @ScopedTimer("env.compute_reward", sync=PROFILE_SYNC_TIMERS)
    def _compute_reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.reward_groups:
            tensordict.set("reward", torch.ones((self.num_envs, 1), device=self.device))
            return tensordict

        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            self.stats[group, "return"].add_(reward)
            if reward_group.enabled:
                tensordict["reward", group] = (
                    reward * self.step_dt if self.mult_dt else reward
                )

        self.stats["episode_len"][:] = self.episode_length_buf.reshape(self.num_envs, 1)
        self.stats["success"][:] = (
            (self.episode_length_buf.reshape(self.num_envs, 1) >= self.max_episode_length * 0.9)
            .float()
        )
        return tensordict

    @ScopedTimer("env.compute_termination", sync=PROFILE_SYNC_TIMERS)
    def _compute_termination(self, tensordict: TensorDictBase) -> TensorDictBase:
        truncated = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        discount = torch.ones((self.num_envs, 1), device=self.device)
        for key, func in self.termination_funcs.items():
            result = func.compute(terminated)
            if isinstance(result, tuple):
                term_value, term_discount = result
            else:
                term_value, term_discount = result, 1.0
            if not func.enabled:
                term_value.zero_()
            if func.is_timeout:
                truncated |= term_value
            else:
                terminated |= term_value
            discount *= term_discount
            self.stats["termination", key] = term_value.float()
        tensordict.set("truncated", truncated)
        tensordict.set("terminated", terminated)
        tensordict.set("done", terminated | truncated)
        tensordict.set("discount", discount)
        return tensordict

    @ScopedTimer("env.compute_observation", sync=PROFILE_SYNC_TIMERS)
    def _compute_observation(self, tensordict: TensorDictBase) -> TensorDictBase:
        [
            group.compute(tensordict, self.timestamp)
            for group in self.observation_groups.values()
        ]
        return tensordict

    @property
    def ground(self):
        if not hasattr(self, "_ground"):
            self._ground = GroundQuery(
                self.terrain_type, self.device, self.ground_mesh
            )
        return self._ground

    @property
    def ground_mesh(self):
        """Warp ground mesh used for ray-based height queries.

        The concrete mesh construction is delegated to the backend-specific
        ``SceneAdapter`` implementations so that this environment stays
        agnostic to how ground geometry is represented per backend.
        """
        return self.scene.ground_mesh

    def get_ground_height_at(self, pos: torch.Tensor) -> torch.Tensor:
        return self.ground.height_at(pos)

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------
    def get_recorder(self, path, enabled: bool = True) -> VideoRecorder:
        """Return a backend-specific video recorder as a context manager.

        Usage:
            with env.get_recorder(\"video.mp4\", enabled) as rec:
                ...
                rec.add_frame()

        For backends with ``rgb_array`` rendering support, this returns a
        streaming recorder. Otherwise, or when ``enabled`` is False, this
        returns a no-op recorder so call sites don't need to branch.
        """
        if not enabled:
            return NullVideoRecorder()
        if self.backend == "isaaclab":
            return IsaacVideoRecorder(self, path, enabled=True)
        if self.backend == "mjlab":
            return RgbArrayVideoRecorder(self, path, enabled=True)
        # Other backends: return a no-op recorder by default.
        return NullVideoRecorder()

    def _set_seed(self, seed: int = -1):
        if self.backend == "isaaclab":
            try:
                import omni.replicator.core as rep

                rep.set_global_seed(seed)
            except ModuleNotFoundError:
                pass
            return torch_utils.set_seed(seed)
        elif self.backend == "mujoco":
            torch.manual_seed(seed)
            np.random.seed(seed)
        elif self.backend == "mjlab":
            torch.manual_seed(seed)
            np.random.seed(seed)
        elif self.backend == "motrix":
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def render(self, mode: str = "human"):
        self.sim.render_gui()
        if mode == "human":
            return None
        if mode == "rgb_array":
            if hasattr(self, "_rgb_annotator"):
                rgb_data = self._rgb_annotator.get_data()
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
                    *rgb_data.shape
                )
                return rgb_data[:, :, :3]
            if self.backend == "mjlab":
                return self.sim.render_rgb_array()
            raise NotImplementedError(
                f"rgb_array mode not supported for backend '{self.backend}'. "
                "Only Isaac and mjlab backends support rgb_array rendering."
            )
        raise NotImplementedError(f"Render mode '{mode}' not supported.")

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["observation_spec"] = self.observation_spec
        state_dict["action_spec"] = self.action_spec
        state_dict["reward_spec"] = self.reward_spec
        return state_dict

    def diagnostics(self) -> dict:
        d = dict(self.extra)
        d.update(self.action_manager.diagnostics())
        return d

    def close(self, *, raise_if_closed: bool = True):
        if not self.is_closed:
            if self.backend == "isaaclab":
                del self.scene
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
            elif self.backend == "mjlab":
                if self.sim.has_gui():
                    self.sim.viewer.close()
                self.sim.close()
            elif self.backend == "motrix":
                self.sim.close()
            super().close(raise_if_closed=raise_if_closed)
