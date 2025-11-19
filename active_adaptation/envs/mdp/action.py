import torch
from typing import Dict, Literal, Tuple, TYPE_CHECKING, List
from typing_extensions import override

from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
from active_adaptation.utils.math import (
    # quat_mul,
    # quat_conjugate,
    # axis_angle_from_quat,
    # quat_inv,
    quat_rotate_inverse,
    quat_rotate,
    yaw_quat,
    clamp_norm
)
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry
from active_adaptation.assets import get_input_joint_indexing

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]
        self.action_buf: torch.Tensor

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass
    
    def process_action(self, action: torch.Tensor):
        pass

    def apply_action(self, action: torch.Tensor, substep: int):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device


class ConcatenatedAction(ActionManager):
    def __init__(self, env, action_managers: List[ActionManager]):
        super().__init__(env)
        self.action_managers: List[ActionManager] = action_managers
        self.action_dims = [action_manager.action_dim for action_manager in self.action_managers]
        self.action_dim = sum(self.action_dims)
    
    @property
    def action_buf(self):
        return torch.cat([action_manager.action_buf for action_manager in self.action_managers], dim=1)

    def apply_action(self, action: torch.Tensor, substep: int):
        actions = torch.split(action, self.action_dims, dim=-1)
        for action_manager, action in zip(self.action_managers, actions):
            action_manager.apply_action(action, substep)
    
    def symmetry_transform(self):
        return SymmetryTransform.cat(
            [action_manager.symmetry_transform() for action_manager in self.action_managers]
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,  # delay in simulation steps
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        input_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env)
        action_scaling = dict(action_scaling)
        self.joint_ids, self.joint_names, self.action_scaling = self.resolve(action_scaling)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        
        # optionally convert the input order to the asset's order
        self.indexing, self.input_joint_names = get_input_joint_indexing(
            input_order=input_order,
            asset_cfg=self.asset.cfg,
            target_joint_names=self.joint_names,
            device=self.device,
        )

        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay
        self.action_dim = len(self.joint_ids)

        self.alpha_range = tuple(alpha_range)
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device) # TODO: permute to (num_envs, 4, action_dim)
            self.action_queue = torch.zeros(self.num_envs, self.max_delay + self.decimation, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    def resolve(self, spec):
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.input_joint_names)
        return transform

    @override
    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        default_joint_pos = self.asset.data.default_joint_pos[env_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha

    @override
    def process_action(self, action: torch.Tensor):
        action = action[:, self.indexing]
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        self.action_queue = torch.where(
            (torch.arange(self.action_queue.shape[1], device=self.device) < self.delay).reshape(self.num_envs, self.action_queue.shape[1], 1),
            self.action_queue,
            action.unsqueeze(1)
        )

    @override
    def apply_action(self, action: torch.Tensor, substep: int):
        # deplay model: each substep, the first action in queue is consumed
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jpos_target = self.default_joint_pos.clone()
        jpos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(jpos_target)


class LegWheel(ActionManager):
    def __init__(
        self,
        env,
        leg_scaling: Dict[str, float],
        wheel_scaling: Dict[str, float]
    ):
        super().__init__(env)
        self.leg_ids, self.leg_names, self.leg_scaling = (
            string_utils.resolve_matching_names_values(
                dict(leg_scaling), self.asset.joint_names
            )
        )
        self.wheel_ids, self.wheel_names, self.wheel_scaling = (
            string_utils.resolve_matching_names_values(
                dict(wheel_scaling), self.asset.joint_names
            )
        )
        self.leg_scaling = torch.tensor(self.leg_scaling, device=self.device)
        self.wheel_scaling = torch.tensor(self.wheel_scaling, device=self.device)

        self.leg_action_dim = len(self.leg_ids)
        self.wheel_action_dim = len(self.wheel_ids)
        self.action_dim = len(self.leg_ids) + len(self.wheel_ids)
        
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device)

    def apply_action(self, tensordict: TensorDictBase, substep: int):
        action = tensordict["action"]
        if substep == 0:
            self.action_buf = self.action_buf.roll(1, dims=1)
            self.action_buf[:, 0] = action
        self.applied_action = self.applied_action.lerp(action, 0.8)
        leg_action, wheel_action = self.applied_action.split([self.leg_action_dim, self.wheel_action_dim], dim=-1)
        leg_pos_target = self.asset.data.default_joint_pos[:, self.leg_ids] + self.leg_scaling * leg_action
        self.asset.set_joint_position_target(leg_pos_target, self.leg_ids)
        wheel_vel_target = self.wheel_scaling * wheel_action
        self.asset.set_joint_velocity_target(wheel_vel_target, self.wheel_ids)

    def symmetry_transform(self):
        return SymmetryTransform.cat([
            joint_space_symmetry(self.asset, self.leg_names),
            joint_space_symmetry(self.asset, self.wheel_names),
        ])
    

class Marker(ActionManager):
    def __init__(self, env, num_markers: int = 1, world_frame: bool = False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.num_markers = num_markers
        self.world_frame = world_frame
        self.has_gui = self.env.sim.has_gui()
        self.action_dim = 3 * self.num_markers

        if self.has_gui and self.env.backend == "isaac":
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Input/Marker",
                    markers={
                        "marker": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                    },
                )
            )
            self.marker.set_visibility(True)

    def process_action(self, action: torch.Tensor):
        if not self.has_gui or action is None:
            return
        
        if not self.world_frame:
            pos = self.asset.data.root_link_pos_w.reshape(self.num_envs, 1, 3)
            quat = self.asset.data.root_link_quat_w.reshape(self.num_envs, 1, 4)
            translations = pos + quat_rotate(quat, action.reshape(self.num_envs, self.num_markers, 3))
        else:
            translations = action
        translations = translations.reshape(self.num_envs * self.num_markers, 3)
        self.marker.visualize(
            translations=translations,
            scales=torch.ones(3, device=self.device).expand_as(translations)
        )

