import torch
import numpy as np
import hydra
import inspect
import re
import warnings
import warp as wp

from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite, 
    Binary,
    Unbounded,
)
from collections import OrderedDict

from abc import abstractmethod
from typing import Dict, cast

import active_adaptation
import active_adaptation.envs.mdp as mdp
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.envs.adapters import wrap_sim, wrap_scene
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if active_adaptation.get_backend() == "isaac":
    import isaacsim.core.utils.torch as torch_utils
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from pxr import UsdGeom, UsdPhysics


EMA_DECAY = 0.99


def parse_name_and_class(s: str):
    pattern = r'^(\w+)\((\w+)\)$'
    match = re.match(pattern, s)
    if match:
        name, cls = match.groups()
        return name, cls
    return s, s


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
            foo = self.compute({}, 0)
            spec = {}
            spec[self.name] = Unbounded(foo[self.name].shape, dtype=foo[self.name].dtype)
            self._spec = Composite(spec, shape=[foo[self.name].shape[0]]).to(foo[self.name].device)
        return self._spec

    def compute(self, tensordict: TensorDictBase, timestamp: int) -> torch.Tensor:
        # torch.compiler.cudagraph_mark_step_begin()
        output = self._compute()
        tensordict[self.name] = output
        return tensordict
    
    # @torch.compile(mode="reduce-overhead")
    def _compute(self) -> torch.Tensor:
        # update only if outdated
        tensors = []
        for obs_key, func in self.funcs.items():
            tensor = func()
            tensors.append(tensor)
        return torch.cat(tensors, dim=-1)
    
    def symmetry_transform(self):
        transforms = []
        for obs_key, func in self.funcs.items():
            transform = func.symmetry_transform()
            transforms.append(transform)
        transform = symmetry_utils.SymmetryTransform.cat(transforms)
        return transform


class _Env(EnvBase):
    def __init__(self, cfg):
        self.backend = active_adaptation.get_backend()
        if self.backend in ("isaac", "mjlab"):
            device = f"cuda:{active_adaptation.get_local_rank()}"
        else:
            device = "cpu"
        self.cfg = cfg
        super().__init__(
            device=device,
            batch_size=[self.cfg.num_envs],
            run_type_checks=False,
        )

        self.setup_scene()
        # Wrap sim and scene with adapters for unified API
        self.sim = wrap_sim(self.sim, self.backend)
        self.scene = wrap_scene(self.scene, self.backend)

        if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
            self.terrain_type = self.scene.terrain.cfg.terrain_type
        else:
            self.terrain_type = "plane"
        self._ground_mesh = None
        
        self.max_episode_length = self.cfg.max_episode_length
        self.step_dt = self.cfg.sim.step_dt
        self.physics_dt = self.sim.get_physics_dt()
        self.decimation = int(self.step_dt / self.physics_dt)
        
        print(f"Step dt: {self.step_dt}, physics dt: {self.physics_dt}, decimation: {self.decimation}")

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_id = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_count = 0

        # parse obs and reward functions
        self.done_spec = Composite(
            done=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            terminated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            truncated=Binary(1, [self.num_envs, 1], dtype=bool, device=self.device),
            shape=[self.num_envs],
            device=self.device
        )

        self.reward_spec = Composite(
            {
                "stats": {
                    "episode_len": Unbounded([self.num_envs, 1]),
                    "success": Unbounded([self.num_envs, 1]),
                },
            },
            shape=[self.num_envs]
        ).to(self.device)

        command_cfg = dict(self.cfg.command)
        class_name = command_cfg.pop("class")
        command = mdp.Command.make(class_name, self, **command_cfg)
        if not command:
            raise ValueError(f"Command class '{class_name}' not found")
        self.command_manager = cast(mdp.Command, command)

        self.addons = OrderedDict()
        self.randomizations = OrderedDict()
        self.observation_funcs: Dict[str, ObsGroup] = OrderedDict()
        self.reward_groups: Dict[str, RewardGroup] = OrderedDict()
        self.input_managers: Dict[str, mdp.ActionManager] = OrderedDict()

        self._startup_callbacks = []
        self._update_callbacks = []
        self._reset_callbacks = []
        self._debug_draw_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []

        self._pre_step_callbacks.append(self.command_manager.step)
        # self._update_callbacks.append(self.command_manager.update)
        self._reset_callbacks.append(self.command_manager.reset)
        self._debug_draw_callbacks.append(self.command_manager.debug_draw)
        
        input_cfg = dict(self.cfg.get("input", {}))
        if not len(input_cfg):
            input_cfg["action"] = self.cfg.action
        
        action_spec = {}
        for input_key, input_cfg in input_cfg.items():
            input_manager: mdp.ActionManager = hydra.utils.instantiate(input_cfg, env=self)
            self.input_managers[input_key] = input_manager
            self._reset_callbacks.append(input_manager.reset)
            self._debug_draw_callbacks.append(input_manager.debug_draw)
            action_spec[input_key] = Unbounded(
                [self.num_envs, input_manager.action_dim],
                device=self.device
            )
        
        self.action_spec = Composite(action_spec, shape=[self.num_envs], device=self.device)
        
        for rand_spec, params in self.cfg.randomization.items():
            rand_name, cls_name = parse_name_and_class(rand_spec)
            rand = mdp.Randomization.make(cls_name, self, **(params if params is not None else {}))
            if not rand:
                continue
            
            rand = cast(mdp.Randomization, rand)
            self.randomizations[rand_name] = rand
            self._startup_callbacks.append(rand.startup)
            self._reset_callbacks.append(rand.reset)
            self._debug_draw_callbacks.append(rand.debug_draw)
            self._pre_step_callbacks.append(rand.step)
            self._update_callbacks.append(rand.update)

        for group_key, params in self.cfg.observation.items():
            funcs = OrderedDict()            
            for obs_spec, kwargs in params.items():
                obs_name, obs_cls_name = parse_name_and_class(obs_spec)
                obs = mdp.Observation.make(obs_cls_name, self, **(kwargs if kwargs is not None else {}))
                if not obs:
                    continue
                obs = cast(mdp.Observation, obs)
                funcs[obs_name] = obs
                self._startup_callbacks.append(obs.startup)
                self._update_callbacks.append(obs.update)
                self._reset_callbacks.append(obs.reset)
                self._debug_draw_callbacks.append(obs.debug_draw)
                self._post_step_callbacks.append(obs.post_step)
            
            self.observation_funcs[group_key] = ObsGroup(group_key, funcs)
        
        for callback in self._startup_callbacks:
            callback()        
       
        reward_spec = Composite({})

        # parse rewards
        self.mult_dt = self.cfg.reward.pop("_mult_dt_", True)

        self._stats_ema = {}

        enabled_groups = 0
        for group_name, func_specs in self.cfg.reward.items():
            print(f"Reward group: {group_name}")
            funcs = OrderedDict()
            self._stats_ema[group_name] = {}
            eval_func = eval(func_specs.pop("_eval_", "lambda *args: sum(args)"))
            enabled = func_specs.pop("_enabled_", True)
            compile = func_specs.pop("_compile_", False)
            enabled_groups += enabled

            for rew_spec, params in func_specs.items():
                rew_name, cls_name = parse_name_and_class(rew_spec)
                reward = mdp.Reward.make(cls_name, self, **(params if params is not None else {}))
                if not reward:
                    continue
                reward = cast(mdp.Reward, reward)
                funcs[rew_name] = reward
                reward_spec["stats", group_name, rew_name] = Unbounded(1, device=self.device)
                self._update_callbacks.append(reward.update)
                self._reset_callbacks.append(reward.reset)
                self._debug_draw_callbacks.append(reward.debug_draw)
                self._pre_step_callbacks.append(reward.step)
                self._post_step_callbacks.append(reward.post_step)
                print(f"\t{rew_name}: \t{reward.weight:.2f}")
                self._stats_ema[group_name][rew_name] = (torch.tensor(0., device=self.device), torch.tensor(0., device=self.device))

            self.reward_groups[group_name] = RewardGroup(self, group_name, funcs, eval_func, enabled, compile)
            reward_spec["stats", group_name, "return"] = Unbounded(1, device=self.device)

        reward_spec["reward"] = Unbounded(max(1, enabled_groups), device=self.device)
        reward_spec["discount"] = Unbounded(1, device=self.device)
        self.reward_spec.update(reward_spec.expand(self.num_envs).to(self.device))
        self.discount = torch.ones((self.num_envs, 1), device=self.device)

        observation_spec = {}
        for group_key, group in self.observation_funcs.items():
            try:
                observation_spec.update(group.spec)
            except Exception as e:
                print(f"Error in computing observation spec for {group_key}: {e}")
                raise e

        self.observation_spec = Composite(
            observation_spec, 
            shape=[self.num_envs],
            device=self.device
        )
        self.observation_spec["episode_id"] = Unbounded(
            [self.num_envs], dtype=torch.long, device=self.device,
        )

        self.termination_funcs = OrderedDict()
        for key, params in self.cfg.termination.items():
            term_name, cls_name = parse_name_and_class(key)
            term = mdp.Termination.make(cls_name, self, **(params if params is not None else {}))
            if not term:
                continue
            term = cast(mdp.Termination, term)
            self.termination_funcs[term_name] = term
            self._update_callbacks.append(term.update)
            self._reset_callbacks.append(term.reset)
            self.reward_spec["stats", "termination", term_name] = Unbounded((self.num_envs, 1), device=self.device)
        
        self.timestamp: int = 0 # global timestamp in steps

        self.stats = self.reward_spec["stats"].zero()
    
        self.input_tensordict = None
        self.extra = {}

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs
    
    @property
    def action_manager(self):
        return self.input_managers["action"]

    @property
    def stats_ema(self):
        result = {}
        for group_key, group in self._stats_ema.items():
            result[group_key] = {}
            for rew_key, (sum, cnt) in group.items():
                result[group_key][rew_key] = (sum / cnt).item()
        return result
    
    def setup_scene(self):
        raise NotImplementedError
    
    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids):
            num_envs = env_ids.numel()
            self.episode_length_buf[env_ids] = 0
            self.episode_id[env_ids] = self.episode_count + torch.arange(num_envs, device=self.device)
            self.episode_count += num_envs

            self._reset_idx(env_ids)
            self.scene.reset(env_ids)
            for callback in self._reset_callbacks:
                callback(env_ids)
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        tensordict.set("episode_id", self.episode_id.clone())
        # self._compute_observation(tensordict)
        return tensordict

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        raise NotImplementedError
    
    def apply_action(self, tensordict: TensorDictBase, substep: int):
        self.input_tensordict = tensordict
        for input_key, input_manager in self.input_managers.items():
            input_manager.apply_action(tensordict.get(input_key), substep)

    def _compute_observation(self, tensordict: TensorDictBase) -> TensorDictBase:
        for group_key, obs_group in self.observation_funcs.items():
            obs_group.compute(tensordict, self.timestamp)
        return tensordict
            
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
        rewards = torch.cat(all_rewards, 1)
        if self.mult_dt:
            rewards *= self.step_dt

        self.stats["episode_len"][:] = self.episode_length_buf.unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.max_episode_length * 0.9).unsqueeze(1).float()
        tensordict.set("reward", rewards)
        return tensordict
    
    def _compute_termination(self) -> TensorDictBase:
        if not self.termination_funcs:
            return torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)
        
        termination = torch.zeros((self.num_envs, 1), dtype=bool, device=self.device)
        for key, func in self.termination_funcs.items():
            t = func.compute(termination)
            termination |= t
            self.stats["termination", key] = t.float()
        return termination

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        with ScopedTimer("simulation", sync=False):
            with ScopedTimer("process_action", sync=False):
                for input_key, input_manager in self.input_managers.items():
                    input_manager.process_action(tensordict.get(input_key))
            for substep in range(self.decimation):
                with ScopedTimer("simulation_pre_step", sync=False):
                    # for asset in self.scene.articulations.values():
                    #     if asset.has_external_wrench:
                    #         asset._external_force_b.zero_()
                    #         asset._external_torque_b.zero_()
                    #         asset.has_external_wrench = False
                    self.apply_action(tensordict, substep)
                    for callback in self._pre_step_callbacks:
                        callback(substep)
                    self.scene.write_data_to_sim()
                with ScopedTimer("simulation_step", sync=False):
                    self.sim.step(render=False)
                with ScopedTimer("simulation_post_step", sync=False):
                    self.scene.update(self.physics_dt)
                    for callback in self._post_step_callbacks:
                        callback(substep)
            with ScopedTimer("update_callbacks", sync=False):
                for callback in self._update_callbacks:
                    callback()
        
        if self.sim.has_gui():
            self.sim.render()
        
        self.episode_length_buf.add_(1)
        self.discount.fill_(1.0)
        self.timestamp += 1
        
        tensordict = TensorDict({}, self.num_envs, device=self.device)

        with ScopedTimer("reward", sync=False):
            tensordict = self._compute_reward(tensordict)
        # Note that command update is a special case
        # it should take place after reward computation
        with ScopedTimer("command", sync=False):
            self.command_manager.update()

        with ScopedTimer("observation", sync=False):
            tensordict = self._compute_observation(tensordict)

        with ScopedTimer("termination", sync=False):
            truncated = (self.episode_length_buf >= self.max_episode_length).unsqueeze(1)
            terminated = self._compute_termination()
            tensordict.set("terminated", terminated)
            tensordict.set("truncated", truncated)
            tensordict.set("done", terminated | truncated)
            tensordict.set("discount", self.discount.clone())
        
        tensordict.set("episode_id", self.episode_id.clone())
        tensordict["stats"] = self.stats.clone()

        if self.sim.has_gui():
            if hasattr(self, "debug_draw"): # isaac only
                self.debug_draw.clear()
            for callback in self._debug_draw_callbacks:
                callback()
        
        return tensordict
    
    @property
    def ground_mesh(self):
        if self._ground_mesh is None:
            if self.backend == "isaac":
                self._ground_mesh = _initialize_warp_meshes("/World/ground", self.device.type)
            elif self.backend == "mujoco":
                self._ground_mesh = wp.Mesh(
                    points=wp.array(self.scene.ground_mesh.vertices, dtype=wp.vec3, device=self.device.type),
                    indices=wp.array(self.scene.ground_mesh.faces.flatten(), dtype=wp.int32, device=self.device.type),
                )
            else:
                raise NotImplementedError
        return self._ground_mesh
        
    def get_ground_height_at(self, pos: torch.Tensor) -> torch.Tensor:
        if self.terrain_type == "plane":
            return torch.zeros(pos.shape[:-1], device=self.device)
        bshape = pos.shape[:-1]
        ray_starts = pos.reshape(-1, 3)
        ray_directions = torch.tensor([0., 0., -1.], device=self.device)
        ray_hits = raycast_mesh(
            ray_starts=ray_starts.reshape(-1, 3),
            ray_directions=ray_directions.expand(bshape.numel(), 3),
            max_dist=100.,
            mesh=self.ground_mesh,
            return_distance=False,
        )[0]
        ray_distance = (ray_hits - ray_starts).norm(dim=-1).nan_to_num(posinf=100.)
        return (ray_starts[:, 2] - ray_distance).to(pos.device).reshape(*bshape)
    
    def _set_seed(self, seed: int = -1):
        if self.backend == "isaac":
            # set seed for replicator
            try:
                import omni.replicator.core as rep

                rep.set_global_seed(seed)
            except ModuleNotFoundError:
                pass
            # set seed for torch and other libraries
            return torch_utils.set_seed(seed)
        elif self.backend == "mujoco":
            torch.manual_seed(seed)
            np.random.seed(seed)
            return seed

    def render(self, mode: str = "human"):
        self.sim.render()
        if mode == "human":
            return None
        elif mode == "rgb_array":
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            raise NotImplementedError

    def state_dict(self):
        sd = super().state_dict()
        sd["observation_spec"] = self.observation_spec
        sd["action_spec"] = self.action_spec
        sd["reward_spec"] = self.reward_spec
        return sd

    def get_extra_state(self) -> dict:
        return dict(self.extra)

    def close(self, *, raise_if_closed: bool = True):
        if not self.is_closed:
            if self.backend == "isaac":
                # destructor is order-sensitive
                del self.scene
                # clear callbacks and instance
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
                # update closing status
            super().close(raise_if_closed=raise_if_closed)


class RewardGroup:
    def __init__(
        self,
        env: _Env,
        name: str,
        funcs: OrderedDict[str, mdp.Reward],
        eval_func,
        enabled: bool = True,
        compile: bool = False,
    ):
        self.env = env
        self.name = name
        self.funcs = funcs
        self.eval_func = eval_func
        self.enabled = enabled
        self.compile = compile
        
        if compile:
            self.compute = torch.compile(self.compute, fullgraph=True)

    def compute(self) -> torch.Tensor:
        all_rewards = []
        for key, func in self.funcs.items():
            reward, count = func()
            self.env.stats[self.name, key].add_(reward)
            ema_sum, ema_cnt = self.env._stats_ema[self.name][key]
            ema_sum.mul_(EMA_DECAY).add_(reward.sum())
            ema_cnt.mul_(EMA_DECAY).add_(count)
            all_rewards.append(reward)
        return self.eval_func(*all_rewards)


def _initialize_warp_meshes(mesh_prim_path, device):
    # check if the prim is a plane - handle PhysX plane as a special case
    # if a plane exists then we need to create an infinite mesh that is a plane
    mesh_prim = sim_utils.get_first_matching_child_prim(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
    )
    # if we did not find a plane then we need to read the mesh
    if mesh_prim is None:
        # obtain the mesh prim
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if valid
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # read the vertices and faces
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    else:
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    # add the warp mesh to the list
    return wp_mesh