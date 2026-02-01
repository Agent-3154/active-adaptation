from __future__ import annotations

import torch
import re
from collections import OrderedDict
from typing import Dict, Literal, Tuple, TYPE_CHECKING, List, Optional
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
from active_adaptation.envs.mdp.base import ActionManager


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class ConcatenatedAction(ActionManager):
    """
    Action manager that concatenates multiple action managers into a single action space.
    
    This class allows combining different action types (e.g., joint position and joint velocity)
    into a single action vector. The action is split and distributed to each sub-action manager
    during processing, and all managers are applied during action application.
    """
    def __init__(self, env, actions: List):
        """Initialize the concatenated action manager.
        
        Args:
            env: The environment instance.
            actions: List of action manager specifications. Each spec should contain a "class"
                key indicating the action manager type, and additional parameters for that manager.
        """
        super().__init__(env)
        self.action_managers: List[ActionManager] = []

        for spec in actions:
            cls = ActionManager.registry[spec.pop("class")]
            self.action_managers.append(cls(self.env, **spec))
        self.action_dims = [action_manager.action_dim for action_manager in self.action_managers]
    
    @property
    def action_dim(self):
        """Total action dimension (sum of all sub-action manager dimensions)."""
        return sum(self.action_dims)

    @property
    def action_buf(self):
        """Concatenated action buffer from all sub-action managers."""
        return torch.cat([action_manager.action_buf for action_manager in self.action_managers], dim=-1)

    @override
    def process_action(self, action: torch.Tensor):
        """Split the concatenated action and process each part with its corresponding manager.
        
        Args:
            action: Concatenated action tensor of shape (num_envs, action_dim).
        """
        actions = torch.split(action, self.action_dims, dim=-1)
        for action_manager, action in zip(self.action_managers, actions):
            action_manager.process_action(action)

    @override
    def apply_action(self, substep: int):
        """Apply actions from all sub-action managers.
        
        Args:
            substep: The current physics substep index.
        """
        for action_manager in self.action_managers:
            action_manager.apply_action(substep)
    
    @override
    def symmetry_transform(self):
        """Get the concatenated symmetry transform from all sub-action managers.
        
        Returns:
            A SymmetryTransform that concatenates transforms from all sub-managers.
        """
        return SymmetryTransform.cat(
            [action_manager.symmetry_transform() for action_manager in self.action_managers]
        )


class JointPosition(ActionManager):
    """
    Action manager for joint position control with delay and smoothing.
    
    Controls joint positions by adding scaled actions to default joint positions.
    Supports action delay (simulating actuator latency) and exponential smoothing
    via an alpha parameter for smoother motion.
    """
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,  # delay in simulation steps
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        input_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        """Initialize the joint position action manager.
        
        Args:
            env: The environment instance.
            action_scaling: Dictionary mapping joint names to scaling factors, or a single
                float value to apply to all joints. Actions are multiplied by this scaling
                before being added to default joint positions.
            max_delay: Maximum delay in simulation steps before actions are applied.
                Actual delay is randomly sampled per environment at reset.
            alpha_range: Tuple of (min, max) values for exponential smoothing factor.
                Higher alpha means less smoothing (more responsive). Sampled per environment at reset.
            input_order: Joint ordering convention for input actions. Can be "isaac", "mujoco",
                or "mjlab". Actions will be reordered to match the asset's internal joint order.
        """
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

    @property
    def action_dim(self):
        """Number of joints controlled by this action manager."""
        return len(self.joint_ids)
    
    def resolve(self, spec):
        """Resolve joint names and scaling values from specification.
        
        Args:
            spec: Dictionary mapping joint name patterns to scaling values.
            
        Returns:
            Tuple of (joint_ids, joint_names, action_scaling) where:
                - joint_ids: List of joint indices
                - joint_names: List of resolved joint names
                - action_scaling: List of scaling values for each joint
        """
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    @override
    def reset(self, env_ids: torch.Tensor):
        """Reset action manager state for specified environments.
        
        Resets delay, action buffers, and samples new alpha values for smoothing.
        Also updates default joint positions with any accumulated offsets.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        default_joint_pos = self.asset.data.default_joint_pos[env_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha

    @override
    def process_action(self, action: Optional[torch.Tensor]):
        """Process incoming action and add it to the action queue with delay.
        
        Reorders action according to input_order, updates the action buffer history,
        and queues the action for application after the delay period.
        
        Args:
            action: Action tensor of shape (num_envs, action_dim) or None to skip processing.
        """
        if action is None:
            return
        action = action[:, self.indexing]
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        self.action_queue = torch.where(
            (torch.arange(self.action_queue.shape[1], device=self.device) < self.delay).reshape(self.num_envs, self.action_queue.shape[1], 1),
            self.action_queue,
            action.unsqueeze(1)
        )

    @override
    def apply_action(self, substep: int):
        """Apply the queued action with exponential smoothing.
        
        Consumes the first action from the queue, applies exponential smoothing,
        and sets the joint position target.
        
        Args:
            substep: The current physics substep index.
        """
        # deplay model: each substep, the first action in queue is consumed
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jpos_target = self.default_joint_pos.clone()
        jpos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(jpos_target)

    @override
    def symmetry_transform(self):
        """Get the symmetry transform for joint space actions.
        
        Returns:
            A SymmetryTransform that handles joint space symmetries (e.g., left/right leg swapping).
        """
        transform = joint_space_symmetry(self.asset, self.input_joint_names)
        return transform


class JointVelocity(ActionManager):
    """
    Action manager for joint velocity control with delay and smoothing.
    
    Controls joint velocities directly. Supports action delay (simulating actuator latency)
    and exponential smoothing via an alpha parameter for smoother motion.
    """
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,  # delay in simulation steps
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        input_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        """Initialize the joint velocity action manager.
        
        Args:
            env: The environment instance.
            action_scaling: Dictionary mapping joint names to scaling factors, or a single
                float value to apply to all joints. Actions are multiplied by this scaling
                before being set as velocity targets.
            max_delay: Maximum delay in simulation steps before actions are applied.
                Actual delay is randomly sampled per environment at reset.
            alpha_range: Tuple of (min, max) values for exponential smoothing factor.
                Higher alpha means less smoothing (more responsive). Sampled per environment at reset.
            input_order: Joint ordering convention for input actions. Can be "isaac", "mujoco",
                or "mjlab". Actions will be reordered to match the asset's internal joint order.
        """
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

        self.alpha_range = tuple(alpha_range)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device) # TODO: permute to (num_envs, 4, action_dim)
            self.action_queue = torch.zeros(self.num_envs, self.max_delay + self.decimation, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
    
    @property
    def action_dim(self):
        """Number of joints controlled by this action manager."""
        return len(self.joint_ids)
    
    def resolve(self, spec):
        """Resolve joint names and scaling values from specification.
        
        Args:
            spec: Dictionary mapping joint name patterns to scaling values.
            
        Returns:
            Tuple of (joint_ids, joint_names, action_scaling) where:
                - joint_ids: List of joint indices
                - joint_names: List of resolved joint names
                - action_scaling: List of scaling values for each joint
        """
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    @override
    def reset(self, env_ids: torch.Tensor):
        """Reset action manager state for specified environments.
        
        Resets delay, action buffers, and samples new alpha values for smoothing.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha

    @override
    def process_action(self, action: Optional[torch.Tensor]):
        """Process incoming action and add it to the action queue with delay.
        
        Reorders action according to input_order, updates the action buffer history,
        and queues the action for application after the delay period.
        
        Args:
            action: Action tensor of shape (num_envs, action_dim) or None to skip processing.
        """
        if action is None:
            return
        action = action[:, self.indexing]
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        self.action_queue = torch.where(
            (torch.arange(self.action_queue.shape[1], device=self.device) < self.delay).reshape(self.num_envs, self.action_queue.shape[1], 1),
            self.action_queue,
            action.unsqueeze(1)
        )

    @override
    def apply_action(self, substep: int):
        """Apply the queued action with exponential smoothing.
        
        Consumes the first action from the queue, applies exponential smoothing,
        and sets the joint velocity target.
        
        Args:
            substep: The current physics substep index.
        """
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jvel_target = self.applied_action * self.action_scaling
        self.asset.set_joint_velocity_target(jvel_target, joint_ids=self.joint_ids)

    @override
    def symmetry_transform(self):
        """Get the symmetry transform for joint space actions.
        
        Returns:
            A SymmetryTransform that handles joint space symmetries (e.g., left/right leg swapping).
        """
        transform = joint_space_symmetry(self.asset, self.input_joint_names)
        return transform


class Marker(ActionManager):
    """
    Action manager for visualizing action targets as markers in the simulation.
    
    This is a visualization-only action manager that displays markers at specified
    positions to help debug and visualize where actions are targeting. Does not
    actually control the robot.
    """
    def __init__(self, env, num_markers: int = 1, body_frame: bool = False):
        """Initialize the marker action manager.
        
        Args:
            env: The environment instance.
            num_markers: Number of markers to visualize.
            body_frame: If True, marker positions are interpreted relative to the robot's
                root link frame. If False, positions are in the environment frame.
        """
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.num_markers = num_markers
        self.body_frame = body_frame
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
        """Process action and update marker positions for visualization.
        
        Converts action positions to world coordinates (either body frame or environment frame)
        and updates the marker visualization.
        
        Args:
            action: Action tensor of shape (num_envs, 3 * num_markers) containing
                marker positions. If body_frame is True, positions are relative to robot root.
                If False, positions are relative to environment origin.
        """
        if not self.has_gui or action is None:
            return
        
        if self.body_frame:
            pos = self.asset.data.root_link_pos_w.reshape(self.num_envs, 1, 3)
            quat = self.asset.data.root_link_quat_w.reshape(self.num_envs, 1, 4)
            translations = pos + quat_rotate(quat, action.reshape(self.num_envs, self.num_markers, 3))
        else:
            # environment frame
            translations = action + self.env.scene.env_origins.unsqueeze(1)
        translations = translations.reshape(self.num_envs * self.num_markers, 3)
        self.marker.visualize(
            translations=translations,
            scales=torch.ones(3, device=self.device).expand_as(translations)
        )


class WriteRootState(ActionManager):
    """
    Directly write the root pose to the simulation for debugging purposes.
    
    This action manager bypasses normal physics and directly sets the robot's root
    pose and velocity. It should only be used for debugging and testing, not for
    normal training or simulation.
    """
    def __init__(self, env):
        """Initialize the write root state action manager.
        
        Args:
            env: The environment instance.
        """
        super().__init__(env)
        # self.asset: Articulation = self.env.scene["robot"]
        self.action_dim = 7 + 6
        self.target_root_pose = None
        self.target_root_velocity = None
    
    def process_action(self, action: torch.Tensor):
        """Process action and store target root pose and velocity.
        
        Args:
            action: Action tensor of shape (num_envs, 13) containing:
                - First 7 elements: root pose (position xyz, quaternion xyzw)
                - Last 6 elements: root velocity (linear xyz, angular xyz)
        """
        self.target_root_pose = action[:, :7]
        self.target_root_pose[:, :3] += self.env.scene.env_origins
        self.target_root_velocity = action[:, 7:]

    @override
    def apply_action(self, substep: int):
        """Directly write root pose and velocity to simulation.
        
        Bypasses normal physics and directly sets the root state. Use only for debugging.
        
        Args:
            substep: The current physics substep index.
        """
        if self.target_root_pose is None:
            return
        self.asset.write_root_pose_to_sim(self.target_root_pose)
        self.asset.write_root_velocity_to_sim(self.target_root_velocity)


class WriteJointPosition(ActionManager):
    """
    Directly write the joint position (with zero velocity) to the simulation for debugging purposes.
    
    This action manager bypasses normal physics and directly sets joint positions
    with zero velocities. It should only be used for debugging and testing, not for
    normal training or simulation.
    """
    def __init__(self, env):
        """Initialize the write joint position action manager.
        
        Args:
            env: The environment instance.
        """
        super().__init__(env)
        # self.asset: Articulation = self.env.scene["robot"]
        self.action_dim = self.asset.data.default_joint_pos.shape[-1]
        self.target_joint_pos = None

    def process_action(self, action: torch.Tensor):
        """Process action and store target joint positions.
        
        Args:
            action: Action tensor of shape (num_envs, num_joints) containing target joint positions.
        """
        self.target_joint_pos = action
    
    @override
    def apply_action(self, substep: int):
        """Directly write joint positions to simulation with zero velocity.
        
        Bypasses normal physics and directly sets joint positions. Joint velocities
        are set to zero. Use only for debugging.
        
        Args:
            substep: The current physics substep index.
        """
        if self.target_joint_pos is None:
            return
        self.asset.set_joint_position_target(self.target_joint_pos)
        self.asset.write_joint_position_to_sim(self.target_joint_pos)
        self.asset.write_joint_velocity_to_sim(torch.zeros_like(self.target_joint_pos))

