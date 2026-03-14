import warnings
from collections import OrderedDict
from typing import Dict, Mapping, Tuple, cast

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs import EnvBase

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.envs.adapters import SimAdapter, SceneAdapter
from active_adaptation.utils.video_recorder import (
    VideoRecorder,
    NullVideoRecorder,
    IsaacVideoRecorder,
)
from active_adaptation.envs.utils import GroundQuery
from active_adaptation.registry import RegistryMixin
from active_adaptation.utils.profiling import ScopedTimer

if active_adaptation.get_backend() == "isaac":
    import isaacsim.core.utils.torch as torch_utils


EMA_DECAY = 0.99


def parse_component_spec(name: str, cfg):
    if cfg is None or not hasattr(cfg, "items"):
        raise ValueError(f"Component '{name}' must be a mapping.")
    kwargs = dict(cfg)
    target = kwargs.pop("_target_", name)
    return name, target, kwargs


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

    @property
    def keys(self):
        return self.funcs.keys()

    @property
    def spec(self):
        if not hasattr(self, "_spec"):
            sample = self.compute({}, 0)
            spec = {
                self.name: Unbounded(
                    sample[self.name].shape,
                    dtype=sample[self.name].dtype,
                )
            }
            self._spec = Composite(spec, shape=[sample[self.name].shape[0]]).to(
                sample[self.name].device
            )
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> TensorDictBase:
        tensordict[self.name] = self._compute()
        return tensordict

    def _compute(self) -> torch.Tensor:
        return torch.cat([func.compute() for func in self.funcs.values()], dim=-1)

    def symmetry_transform(self):
        transforms = [
            func.symmetry_transform().to(func.device) for func in self.funcs.values()
        ]
        return symmetry_utils.SymmetryTransform.cat(transforms)


class RewardGroup:
    """Group of reward terms with internal EMA tracking for logging."""

    def __init__(
        self,
        env: "_EnvBase",
        name: str,
        funcs: OrderedDict[str, mdp.Reward],
        enabled: bool = True,
        compile: bool = False,
    ):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.enabled = enabled
        self.compile = compile

        self.enabled_rewards = sum(func.enabled for func in funcs.values())
        self.rew_buf = torch.zeros(
            env.num_envs, self.enabled_rewards, device=env.device
        )
        # EMA state per reward term (sum, count) for logging
        self._ema: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            key: (
                torch.zeros(1, device=env.device),
                torch.zeros(1, device=env.device),
            )
            for key in funcs.keys()
        }
        if compile:
            self.compute = torch.compile(self.compute, fullgraph=True)

    def compute(self) -> torch.Tensor:
        rewards = []
        for key, func in self.funcs.items():
            reward, count = func.compute()
            self.env.stats[self.name, key].add_(reward)
            ema_sum, ema_cnt = self._ema[key]
            ema_sum.mul_(EMA_DECAY).add_(reward.sum())
            cnt = count.item() if torch.is_tensor(count) else count
            ema_cnt.mul_(EMA_DECAY).add_(cnt)
            if func.enabled:
                rewards.append(reward)
        if len(rewards):
            self.rew_buf[:] = torch.cat(rewards, 1)
        return self.rew_buf.sum(dim=1, keepdim=True)

    def get_ema_stats(self) -> Dict[str, float]:
        """Return EMA mean per reward term (for logging)."""
        result = {}
        for key, (ema_sum, ema_cnt) in self._ema.items():
            cnt = ema_cnt.clamp(min=1e-8)
            result[key] = (ema_sum / cnt).item()
        return result


class _EnvBase(EnvBase, RegistryMixin):
    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(
            device=device,
            batch_size=[cfg.num_envs],
            run_type_checks=False,
        )
        self.backend = active_adaptation.get_backend()
        self.cfg = cfg
        self.headless = headless

        self._setup_simulation()
        self._setup_mdp_managers()
        self._build_tensor_specs()

        [callback() for callback in self._startup_callbacks]

        self.timestamp: int = 0
        self.stats: TensorDict = self.reward_spec["stats"].zero()
        self.input_tensordict = None
        self.extra = {}

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------
    def _setup_simulation(self):
        self.terrain_type = None
        self.setup_scene()
        self.sim = cast(SimAdapter, self.sim)
        self.scene = cast(SceneAdapter, self.scene)
        if self.terrain_type is None:
            warnings.warn(
                "Terrain type is not set. Please check if the scene is properly initialized."
            )
        self.max_episode_length = int(self.cfg.max_episode_length)
        self.step_dt = float(self.cfg.sim.step_dt)
        self.physics_dt = float(self.sim.get_physics_dt())
        self.decimation = int(self.step_dt / self.physics_dt)

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_count = 0
        self.current_iter = 0

    def _setup_mdp_managers(self):
        self.randomizations: Mapping[str, mdp.Randomization] = OrderedDict()
        self.observation_funcs: Mapping[str, ObsGroup] = OrderedDict()
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
        if class_name is None:
            raise ValueError("Command config must provide `_target_`.")
        command = mdp.Command.make(class_name, self, **command_cfg)
        if not command:
            raise ValueError(f"Command class '{class_name}' not found")
        self.command_manager = cast(mdp.Command, command)
        self._pre_step_callbacks.append(self.command_manager.pre_step)
        self._reset_callbacks.append(self.command_manager.reset)
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)

        # MDP: input managers
        for input_name, input_cfg in dict(self.cfg.get("input", {})).items():
            _, input_cls_name, input_kwargs = parse_component_spec(
                input_name, input_cfg
            )
            input_cls = mdp.Action.registry[input_cls_name]
            input_manager = cast(mdp.Action, input_cls(self, **input_kwargs))
            self.input_managers[input_name] = input_manager
            self._reset_callbacks.append(input_manager.reset)
            self._debug_draw_callbacks.append(input_manager.debug_draw)

        # MDP: randomizations
        for rand_name, rand_cfg in self.cfg.get("randomization", {}).items():
            rand_name, cls_name, rand_kwargs = parse_component_spec(rand_name, rand_cfg)
            rand = mdp.Randomization.make(cls_name, self, **rand_kwargs)
            if not rand:
                continue
            rand = cast(mdp.Randomization, rand)
            self.randomizations[rand_name] = rand
            self._add_mdp_component(rand)

        # MDP: observations
        for group_name, group_cfg in self.cfg.observation.items():
            funcs = OrderedDict()
            for obs_name, obs_cfg in group_cfg.items():
                obs_name, obs_cls_name, obs_kwargs = parse_component_spec(
                    obs_name, obs_cfg
                )
                obs = mdp.Observation.make(obs_cls_name, self, **obs_kwargs)
                if not obs:
                    continue
                obs = cast(mdp.Observation, obs)
                funcs[obs_name] = obs
                self._add_mdp_component(obs)
            self.observation_funcs[group_name] = ObsGroup(group_name, funcs)

        # MDP: rewards
        reward_cfg = dict(self.cfg.reward)
        self.mult_dt = reward_cfg.pop("_mult_dt_", True)
        for group_name, group_cfg in reward_cfg.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()

            group_cfg = dict(group_cfg)
            enabled = group_cfg.pop("_enabled_", True)
            compile = group_cfg.pop("_compile_", False)
            self._enabled_reward_groups += int(enabled)

            for rew_name, rew_cfg in group_cfg.items():
                rew_name, cls_name, rew_kwargs = parse_component_spec(rew_name, rew_cfg)
                reward = mdp.Reward.make(cls_name, self, **rew_kwargs)
                if not reward:
                    continue
                reward = cast(mdp.Reward, reward)
                funcs[rew_name] = reward
                self._add_mdp_component(reward)
                print(f"\t{rew_name}: \t{reward.weight:.2f}")

            self.reward_groups[group_name] = RewardGroup(
                self, group_name, funcs, enabled, compile
            )

        # MDP: terminations
        for term_name, term_cfg in self.cfg.get("termination", {}).items():
            term_name, cls_name, term_kwargs = parse_component_spec(term_name, term_cfg)
            term = mdp.Termination.make(cls_name, self, **term_kwargs)
            if not term:
                continue
            term = cast(mdp.Termination, term)
            self.termination_funcs[term_name] = term
            self._add_mdp_component(term)

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
            for group in self.observation_funcs.values()
        ]
        self.observation_spec = Composite(
            observation_spec, shape=[self.num_envs], device=self.device
        )
        self.observation_spec["episode_id"] = Unbounded(
            [self.num_envs], dtype=torch.long, device=self.device
        )

        reward_spec = Composite(
            {
                "stats": {
                    "episode_len": Unbounded([self.num_envs, 1]),
                    "success": Unbounded([self.num_envs, 1]),
                }
            },
            shape=[self.num_envs],
        ).to(self.device)
        reward_spec_extensions = Composite({})

        for group_name, reward_group in self.reward_groups.items():
            for rew_name in reward_group.funcs.keys():
                reward_spec_extensions["stats", group_name, rew_name] = Unbounded(
                    1, device=self.device
                )
            reward_spec_extensions["stats", group_name, "return"] = Unbounded(
                1, device=self.device
            )

        for term_name in self.termination_funcs.keys():
            reward_spec_extensions["stats", "termination", term_name] = Unbounded(
                1, device=self.device
            )

        reward_spec_extensions["reward"] = Unbounded(
            self._enabled_reward_groups, device=self.device
        )
        reward_spec_extensions["discount"] = Unbounded(1, device=self.device)
        reward_spec.update(reward_spec_extensions.expand(self.num_envs).to(self.device))
        self.reward_spec = reward_spec

    def _add_mdp_component(self, component: mdp.MDPComponent):
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

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids):
            num_envs = env_ids.numel()
            self.episode_length_buf[env_ids] = 0
            self.episode_id[env_ids] = self.episode_count + torch.arange(
                num_envs, device=self.device
            )
            self.episode_count += num_envs

            self._reset_idx(env_ids)
            self.scene.reset(env_ids)
            [callback(env_ids) for callback in self._reset_callbacks]

        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        tensordict.set("episode_id", self.episode_id.clone())
        return tensordict

    def _reset_idx(self, env_ids: torch.Tensor):
        init_state = self.command_manager.sample_init(env_ids)
        if not isinstance(init_state, dict):
            init_state = {"robot": init_state}
        for key, value in init_state.items():
            if value is not None:
                self.scene.articulations[key].write_root_state_to_sim(
                    value, env_ids=env_ids
                )
        self.stats[env_ids] = 0.0

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        with ScopedTimer("simulation", sync=False):
            with ScopedTimer("process_action", sync=False):
                for input_key, input_manager in self.input_managers.items():
                    if (action := tensordict.get(input_key)) is not None:
                        input_manager.process_action(action)

            for substep in range(self.decimation):
                with ScopedTimer("simulation_pre_step", sync=False):
                    self.scene.zero_external_wrenches()
                    self._apply_action(substep)
                    [callback(substep) for callback in self._pre_step_callbacks]
                    self.scene.write_data_to_sim()
                with ScopedTimer("simulation_step", sync=False):
                    self.sim.step(render=False)
                with ScopedTimer("simulation_post_step", sync=False):
                    self.scene.update(self.physics_dt)
                    [callback(substep) for callback in self._post_step_callbacks]

            with ScopedTimer("update_callbacks", sync=False):
                [callback() for callback in self._update_callbacks]

        if self.sim.has_gui():
            self.sim.render()

        self.episode_length_buf.add_(1)
        self.timestamp += 1

        tensordict = TensorDict({}, self.num_envs, device=self.device)

        with ScopedTimer("reward", sync=False):
            tensordict = self._compute_reward(tensordict)
        with ScopedTimer("termination", sync=False):
            tensordict = self._compute_termination(tensordict)
        with ScopedTimer("command", sync=False):
            self.command_manager.update()
        with ScopedTimer("observation", sync=False):
            tensordict = self._compute_observation(tensordict)

        tensordict.set("episode_id", self.episode_id.clone())
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui():
            if self.backend == "isaac":
                self.debug_draw.clear()
            elif self.backend == "mjlab":
                self.sim.viewer.clear()
            [callback() for callback in self._debug_draw_callbacks]
            if self.backend == "mjlab":
                self.sim.viewer.update()

        return tensordict

    def _apply_action(self, substep: int):
        [
            input_manager.apply_action(substep)
            for input_manager in self.input_managers.values()
        ]

    def _compute_reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.reward_groups:
            tensordict.set("reward", torch.ones((self.num_envs, 1), device=self.device))
            return tensordict

        all_rewards = []
        for group, reward_group in self.reward_groups.items():
            reward = reward_group.compute()
            self.stats[group, "return"].add_(reward)
            if reward_group.enabled:
                all_rewards.append(reward)
        rewards = torch.cat(all_rewards, dim=1)
        if self.mult_dt:
            rewards *= self.step_dt

        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (
            (self.episode_length_buf >= self.max_episode_length * 0.9)
            .unsqueeze(1)
            .float()
        )
        tensordict.set("reward", rewards)
        return tensordict

    def _compute_termination(self, tensordict: TensorDictBase) -> TensorDictBase:
        truncated = self.episode_length_buf[:, None] >= self.max_episode_length
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

    def _compute_observation(self, tensordict: TensorDictBase) -> TensorDictBase:
        [
            group.compute(tensordict, self.timestamp)
            for group in self.observation_funcs.values()
        ]
        return tensordict

    @property
    def ground(self):
        if not hasattr(self, "_ground"):
            self._ground = GroundQuery(
                self.scene, self.backend, self.terrain_type, self.device
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

        For non-Isaac backends, or when ``enabled`` is False, this returns a
        no-op recorder so call sites don't need to branch.
        """
        if not enabled:
            return NullVideoRecorder()
        if self.backend == "isaac":
            return IsaacVideoRecorder(self, path, enabled=True)
        # Other backends: return a no-op recorder by default.
        return NullVideoRecorder()

    def _set_seed(self, seed: int = -1):
        if self.backend == "isaac":
            try:
                import omni.replicator.core as rep

                rep.set_global_seed(seed)
            except ModuleNotFoundError:
                pass
            return torch_utils.set_seed(seed)
        if self.backend == "mujoco":
            torch.manual_seed(seed)
            np.random.seed(seed)
            return seed

    def render(self, mode: str = "human"):
        self.sim.render()
        if mode == "human":
            return None
        if mode == "rgb_array":
            if hasattr(self, "_rgb_annotator"):
                rgb_data = self._rgb_annotator.get_data()
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
                    *rgb_data.shape
                )
                return rgb_data[:, :, :3]
            raise NotImplementedError(
                f"rgb_array mode not supported for backend '{self.backend}'. "
                "Only Isaac backend supports rgb_array rendering."
            )
        raise NotImplementedError(f"Render mode '{mode}' not supported.")

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["observation_spec"] = self.observation_spec
        state_dict["action_spec"] = self.action_spec
        state_dict["reward_spec"] = self.reward_spec
        return state_dict

    def get_extra_state(self) -> dict:
        return dict(self.extra)

    def close(self, *, raise_if_closed: bool = True):
        if not self.is_closed:
            if self.backend == "isaac":
                del self.scene
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
            elif self.backend == "mjlab" and self.sim.has_gui():
                self.sim.viewer.close()
            super().close(raise_if_closed=raise_if_closed)

