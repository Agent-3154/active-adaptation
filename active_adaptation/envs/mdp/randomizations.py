import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple, Optional

import active_adaptation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

import isaaclab.utils.string as string_utils


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


if active_adaptation.get_backend() == "isaac":
    from isaaclab.actuators import DCMotor, ImplicitActuator
    from active_adaptation.envs.actuator import HybridActuator

from .base import Randomization

RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


class motor_params(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self,
        env,
        stiffness_range: Optional[NestedRangeType] = None,
        damping_range: Optional[NestedRangeType] = None,
        armature_range: Optional[NestedRangeType] = None,
    ):
        super().__init__(env),
        self.asset: Articulation = self.env.scene["robot"]
        self.indices = {}
        self.ranges = {}
        self.write_func = {}

        if stiffness_range is not None:
            self.stiffness_range = dict(stiffness_range)
            ids, _, value = string_utils.resolve_matching_names_values(self.stiffness_range, self.asset.joint_names)
            default = self.asset.data.joint_stiffness[0, ids]
            low, high = (torch.tensor(value, device=self.device) * default.unsqueeze(1)).unbind(1)
            self.indices["stiffness"] = torch.tensor(ids, device=self.device)
            self.ranges["stiffness"] = (low, high - low)
            self.write_func["stiffness"] = self.asset.write_joint_stiffness_to_sim
        
        if damping_range is not None:
            self.damping_range = dict(damping_range)
            ids, _, value = string_utils.resolve_matching_names_values(self.damping_range, self.asset.joint_names)
            default = self.asset.data.joint_damping[0, ids]
            low, high = (torch.tensor(value, device=self.device) * default.unsqueeze(1)).unbind(1)
            self.indices["damping"] = torch.tensor(ids, device=self.device)
            self.ranges["damping"] = (low, high - low)
            self.write_func["damping"] = self.asset.write_joint_damping_to_sim

        if armature_range is not None:
            self.armature_range = dict(armature_range)
            ids, _, value = string_utils.resolve_matching_names_values(self.armature_range, self.asset.joint_names)
            low, high = torch.tensor(value, device=self.device).unbind(1)
            self.indices["armature"] = torch.tensor(ids, device=self.device)
            self.ranges["armature"] = (low, high - low)
            self.write_func["armature"] = self.asset.write_joint_armature_to_sim
        
    def reset(self, env_ids):
        for key, indices in self.indices.items():
            low, range = self.ranges[key]
            values = torch.rand(len(env_ids), len(indices), device=self.device) * range + low
            self.write_func[key](values, indices, env_ids)


class motor_params_armature(Randomization):
    def __init__(self, env, stiffness_range, damping_range, armature_range):
        super().__init__(env),
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness_range = dict(stiffness_range)
        self.damping_range = dict(damping_range)
        self.armature_range = dict(armature_range)

        ids, _, value = string_utils.resolve_matching_names_values(self.stiffness_range, self.asset.joint_names)
        self.stiffness_id = torch.tensor(ids, device=self.device)
        self.stiffness_default = self.asset.data.joint_stiffness[0, self.stiffness_id]
        low, high = (torch.tensor(value, device=self.device) * self.stiffness_default.unsqueeze(1)).unbind(1)
        self.stiffness_low = low
        self.stiffness_scale = high - low

        ids, _, value = string_utils.resolve_matching_names_values(self.damping_range, self.asset.joint_names)
        self.damping_id = torch.tensor(ids, device=self.device)
        self.damping_default = self.asset.data.joint_damping[0, self.damping_id]
        low, high = (torch.tensor(value, device=self.device) * self.damping_default.unsqueeze(1)).unbind(1)
        self.damping_low = low
        self.damping_scale = high - low

        ids, _, value = string_utils.resolve_matching_names_values(self.armature_range, self.asset.joint_names)
        self.armature_id = torch.tensor(ids, device=self.device)
        self.armature_default = self.asset.data.joint_armature[0, self.armature_id]
        low, high = (torch.tensor(value, device=self.device) * self.armature_default.unsqueeze(1)).unbind(1)
        self.armature_low = low
        self.armature_scale = high - low
    
    def reset(self, env_ids):
        stiffness = torch.rand(len(env_ids), len(self.stiffness_id), device=self.device) * self.stiffness_scale + self.stiffness_low
        self.asset.write_joint_stiffness_to_sim(stiffness, self.stiffness_id, env_ids)

        damping = torch.rand(len(env_ids), len(self.damping_id), device=self.device) * self.damping_scale + self.damping_low
        self.asset.write_joint_damping_to_sim(damping, self.damping_id, env_ids)

        armature = torch.rand(len(env_ids), len(self.armature_id), device=self.device) * self.armature_scale + self.armature_low
        self.asset.write_joint_armature_to_sim(armature, self.armature_id, env_ids)



class random_motor_failure(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self,
        env,
        actuator_name: str,
        joint_names: str,
        failure_prob: float = 0.2,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.motors: DCMotor = self.asset.actuators[actuator_name]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names, self.motors.joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.failure_prob = failure_prob
        assert not hasattr(self.motors, "motor_failure")
        self.motor_failure = self.motors.motor_failure = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        logging.info(f"Randomly disable one joint from {self.joint_names} with prob. {self.failure_prob}.")
        self.failure_prob = failure_prob

        # hard-coded
        self._body_ids = self.asset.find_bodies(".*calf.*")[0]
        
    def reset(self, env_ids: torch.Tensor):
        self.motor_failure[env_ids] = -1.0
        with torch.device(self.device):
            env_ids = env_ids[torch.rand(len(env_ids)) < self.failure_prob]
            i = torch.randint(0, len(self.joint_ids), env_ids.shape)
            joint_id = self.joint_ids[i]
        self.motors.stiffness[env_ids, joint_id] = 0.02
        self.motors.damping[env_ids, joint_id] = 0.02
        self.motor_failure[env_ids, i] = 1.0

    def debug_draw(self):
        x = self.asset.data.body_link_pos_w[:, self._body_ids]
        x = x[self.motor_failure > 0.]
        self.env.debug_draw.point(x, color=(0.1, 1.0, 0.1, 0.8), size=20)


class perturb_body_materials(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool=False
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.homogeneous = homogeneous
        
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0,] + num_shapes_per_body)
        self.shape_ids = torch.cat([
            torch.arange(cumsum[i], cumsum[i+1]) 
            for i in self.body_ids
        ])
        self.num_buckets = 64
        self.static_friction_buckets = torch.linspace(*self.static_friction_range, self.num_buckets)
        self.dynamic_friction_buckets = torch.linspace(*self.dynamic_friction_range, self.num_buckets)
        self.restitution_buckets = torch.linspace(*self.restitution_range, self.num_buckets)

    def startup(self):
        logging.info(f"Randomize body materials of {self.body_names} upon startup.")

        materials = self.asset.root_physx_view.get_material_properties().clone()
        if self.homogeneous:
            shape = (self.num_envs, 1)
        else:
            shape = (self.num_envs, len(self.shape_ids))
        materials[:, self.shape_ids, 0] = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        materials[:, self.shape_ids, 2] = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]

        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)
        self.asset.data.body_materials = materials.to(self.device)


class rand_body_materials(Randomization):
    
    supported_backends = ("isaac",)

    def __init__(
        self,
        env,
        static_friction_range: NestedRangeType,
        dynamic_friction_range: NestedRangeType,
        restitution_range: NestedRangeType,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        shape_start_id = np.cumsum([0,] + num_shapes_per_body)
        
        def parse(body_ids, values):
            shape_ids = []
            ranges = []
            for body_id, value in zip(body_ids, values):
                body_shape_ids = torch.arange(shape_start_id[body_id], shape_start_id[body_id+1])
                shape_ids.append(body_shape_ids)
                ranges.extend([value] * len(body_shape_ids))
            return torch.cat(shape_ids), torch.as_tensor(ranges).T

        body_ids, body_names, values = string_utils.resolve_matching_names_values(dict(static_friction_range), self.asset.body_names)
        self.static_friction_shape_ids, self.static_friction_range = parse(body_ids, values)
        
        body_ids, body_names, values = string_utils.resolve_matching_names_values(dict(dynamic_friction_range), self.asset.body_names)
        self.dynamic_friction_shape_ids, self.dynamic_friction_range = parse(body_ids, values)

        body_ids, body_names, values = string_utils.resolve_matching_names_values(dict(restitution_range), self.asset.body_names)
        self.restitution_shape_ids, self.restitution_range = parse(body_ids, values)

        self.default_materials = self.asset.root_physx_view.get_material_properties()
    
    def startup(self):
        materials = self.default_materials.clone()
        materials[:, self.static_friction_shape_ids, 0] = sample_uniform(len(self.static_friction_shape_ids), *self.static_friction_range)
        materials[:, self.dynamic_friction_shape_ids, 1] = sample_uniform(len(self.dynamic_friction_shape_ids), *self.dynamic_friction_range)
        materials[:, self.restitution_shape_ids, 2] = sample_uniform(len(self.restitution_shape_ids), *self.restitution_range)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)


class perturb_body_mass(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        self.mass_ranges = torch.tensor(values)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        masses = self.asset.data.default_mass.clone()
        inertias = self.asset.data.default_inertia.clone()
        print(f"Default masses: {masses[0]}")
        scale = uniform(
            self.mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
            self.mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
        )
        masses[:, self.body_ids] *= scale
        inertias[:, self.body_ids] *= scale.unsqueeze(-1)
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_masses(masses, indices)
        self.asset.root_physx_view.set_inertias(inertias, indices)
        assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)


class perturb_body_com(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self, env, body_names, pos_range = (-0.05, 0.05)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        
        self.pos_ranges = torch.tensor(pos_range)
        print(self.body_names)

    def startup(self):
        logging.info(f"Randomize body CoM of {self.body_names} upon startup.")
        rand_sample = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        coms = self.asset.root_physx_view.get_coms().clone()
        rand_sample[:, :, :] = uniform(
            self.pos_ranges[0].expand_as(coms[:, self.body_ids, :3]),
            self.pos_ranges[1].expand_as(coms[:, self.body_ids, :3])
        )
        rand_sample[:, :, 0] *= 0.5
        coms[:, self.body_ids, :3] += rand_sample.to('cpu')
        indices = torch.arange(self.asset.num_instances)
        self.asset.root_physx_view.set_coms(coms, indices)
        assert torch.allclose(self.asset.root_physx_view.get_coms(), coms)


class push_by_setting_velocity(Randomization):
    def __init__(self, env, velocity_range=(-0.5, 0.5), min_interval=200):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.velocity_range = velocity_range
        self.min_interval = min_interval
        
        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, 1)
            self.push_velocity = torch.zeros(self.env.num_envs, 6)
    
    def reset(self, env_ids: torch.Tensor):
        self.push_velocity[env_ids] = 0.
        self.last_push[env_ids] = 0.
    
    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1)
            i = torch.rand(self.env.num_envs, 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval)
            self.last_push = torch.where(i, t, self.last_push)
            vel_w = self.asset.data.root_vel_w
            push_velocity = torch.zeros_like(self.push_velocity)
            push_velocity[:, 0].uniform_(*self.velocity_range) * 0.5
            push_velocity[:, 1].uniform_(*self.velocity_range) * 0.5
            push_velocity[:, 2].uniform_(*self.velocity_range) * 0.2
            push_velocity[:, 3].uniform_(*self.velocity_range) * 0.52
            push_velocity[:, 4].uniform_(*self.velocity_range) * 0.52
            push_velocity[:, 5].uniform_(*self.velocity_range) * 0.78
            self.push_velocity = torch.where(i, vel_w + push_velocity, vel_w)
        self.asset.write_root_velocity_to_sim(self.push_velocity)


class reset_joint_states_uniform(Randomization):
    def __init__(
        self,
        env,
        pos_ranges: Dict[str, tuple],
        vel_ranges: Dict[str, tuple]=None,
        rel: bool=False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.rel = rel

        self.joint_ids, self.joint_names, self.pos_ranges = string_utils.resolve_matching_names_values(
            dict(pos_ranges), self.asset.joint_names
        )
        self.pos_ranges = torch.as_tensor(self.pos_ranges, device=self.device).unbind(-1)
        if vel_ranges is not None:
            _, _, self.vel_ranges = string_utils.resolve_matching_names_values(
                dict(vel_ranges), self.asset.joint_names
            )
            self.vel_ranges = torch.as_tensor(self.vel_ranges, device=self.device).unbind(-1)
        else:
            self.vel_ranges = None
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].float()
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids].float()
        self.joint_limits = self.asset.data.joint_pos_limits[0, self.joint_ids].float().unbind(-1)

    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), len(self.joint_ids))
        init_pos = sample_uniform(shape, *self.pos_ranges, self.device)
        if self.rel:
            init_pos += self.default_joint_pos[env_ids]
        if self.vel_ranges is not None:
            init_vel = sample_uniform(shape, *self.vel_ranges, self.device)
        else:
            init_vel = torch.zeros(shape, device=self.device)
        init_vel += self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos.clamp(*self.joint_limits), 
            init_vel, self.joint_ids, env_ids #.unsqueeze(1)
        )


class reset_joint_states_scale(Randomization):
    def __init__(self, env, pos_scales: Dict[str, tuple]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.joint_ids = []
        self.pos_scales = []
        for joint_name, (low, high) in pos_scales.items():
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_ids.extend(joint_ids)
            self.pos_scales.append(torch.tensor([low, high], device=self.env.device).expand(len(joint_ids), 2))
            print(f"Reset {joint_names} to scales of U({low}, {high})")
        self.pos_scales = torch.cat(self.pos_scales, 0).unbind(1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
    
    def reset(self, env_ids: torch.Tensor):
        init_pos = random_scale(
            self.default_joint_pos[env_ids], 
            *self.pos_scales, 
            self.env.device
        )[0]
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos, init_vel, self.joint_ids, env_ids #.unsqueeze(1)
        )


class push(Randomization):
    def __init__(self, env, body_names, force_range = (0.2, 0.9), min_interval=100, decay: float=0.9):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.force_range = force_range
        self.min_interval = min_interval
        self.decay = decay
        
        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, len(self.body_indices), 1)
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.last_push[env_ids] = 0.

    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
            i = torch.rand(self.env.num_envs, len(self.body_indices), 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval)
            self.last_push = torch.where(i, t, self.last_push)

            push_forces = torch.zeros_like(self.forces)
            push_forces[:, :, 0].uniform_(*self.force_range)
            push_forces[:, :, 1].uniform_(*self.force_range)
            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * self.decay)
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, .4, 1.)
        )
        
    
class drag(Randomization):
    def __init__(self, env, body_names, drag_range=(0.0, 0.1)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.drag_coeffs = sample_uniform((self.num_envs, self.num_bodies, 1), *drag_range, self.device).expand(self.num_envs, self.num_bodies, 3)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81

        with torch.device(self.env.device):
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.

    def step(self, substep):
        lin_vel = self.asset.data.body_lin_vel_w[:, self.body_indices]
        drag_forces = - lin_vel * self.drag_coeffs
        self.forces = drag_forces * self.default_mass_total
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total * 100,
            color=(0.6, 0.8, 0.6, 1.)
        )

class stumble(Randomization):
    def __init__(
        self, 
        env,
        body_names: str,
        stumble_height: float=0.05,
        friction_range=(0.0, 0.2),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.num_feet = len(self.body_ids)

        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)
        self.stumble_height = stumble_height
        self.friction_range = friction_range
        self.friction_coef = torch.zeros(self.num_envs, 1, 1, device=self.device)
    
    def startup(self):
        self.feet_height: torch.Tensor = self.asset.data.feet_height

    def reset(self, env_ids: torch.Tensor):
        friction = torch.empty(len(env_ids), 1, 1, device=self.device)
        friction.uniform_(*self.friction_range)
        self.friction_coef[env_ids] = friction

    def step(self, substep):
        # feet_height = self.asset.data.feet_height_map.mean(-1).reshape(-1)
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.body_ids]
        feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
        stumble_prob = ((self.stumble_height - self.feet_height) / self.stumble_height).clamp(0., 1.)
        self.forces_w = - self.friction_coef * feet_lin_vel_w / self.env.physics_dt
        self.forces_w[..., 2] = 0.
        friction_forces = torch.where(
            (torch.rand_like(self.feet_height) < stumble_prob).unsqueeze(-1),
            quat_rotate_inverse(feet_quat_w, self.forces_w),
            torch.zeros(self.num_envs, self.num_feet, 3, device=self.env.device)
        )
        forces_b = self.asset._external_force_b.clone()
        torques_b = self.asset._external_torque_b.clone()
        forces_b[:, self.body_ids] += friction_forces
        self.asset.set_external_force_and_torque(forces_b, torques_b)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_ids],
            self.forces_w * self.env.physics_dt,
            color=(1., 0.6, 0., 1.)
        )


class pull(Randomization):
    def __init__(
        self, 
        env,
        drag_prob: float = 0.2,
        drag_range=(0.0, 0.2)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.drag_prob = drag_prob
        self.drag_range = drag_range
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.device) * 9.81
        
        with torch.device(self.device):
            self.forces = torch.zeros(self.num_envs, 3)
            self.axis = torch.zeros(self.num_envs, 3)
            self.apply_drag = torch.zeros(self.num_envs, 1, dtype=bool)
            self.drag_magnitude = torch.zeros(self.num_envs, 1)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        
        # pull direction
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        axis = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
        self.axis[env_ids] = axis

        drag_magnitude = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.drag_range)
        self.drag_magnitude[env_ids] = drag_magnitude * self.default_mass_total
        self.apply_drag[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) < self.drag_prob)
    
    def update(self):
        pass

    def step(self, substep):
        force =  self.axis * self.drag_magnitude
        self.forces[:] = torch.where(self.apply_drag, force, torch.zeros_like(self.forces))
        self.asset.set_external_force_and_torque(
            quat_rotate_inverse(self.asset.data.root_link_quat_w, self.forces).unsqueeze(1), 
            torch.zeros_like(force).unsqueeze(1), [0])

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w, 
            self.forces / self.default_mass_total, 
            color=(0.6, 0.8, 0.6, 1.)
        )


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset


class spring_grf(Randomization):
    def __init__(self, env, feet_names: str = ".*_foot", thres_range = (0.1, 0.2), kp_range = (200, 300)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.thres_range = thres_range
        self.kp_range = kp_range

        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.kp = torch.zeros(self.num_envs, 4, device=self.device)
        self.thres = torch.zeros(self.num_envs, 4, device=self.device)
        self.forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.flag = torch.zeros(self.num_envs, 4, dtype=bool, device=self.device)
        self.axis = torch.zeros(self.num_envs, 4, 3, device=self.device)

    def update(self):
        resample = (self.env.episode_length_buf % 100 == 0).unsqueeze(1) # [num_envs, 1]
        self.flag = torch.where(resample, torch.rand(self.flag.shape, device=self.device) < 0.2, self.flag)
        self.kp = torch.where(resample, uniform_like(self.kp, *self.kp_range), self.kp)
        self.thres = torch.where(resample, uniform_like(self.thres, *self.thres_range), self.thres)
        axis = torch.zeros(self.num_envs, 4, 3, device=self.device)
        axis[:, :, 1].uniform_(-0.3, 0.3)
        axis[:, :, 0].uniform_(-0.3, 0.3)
        axis[:, :, 2] = 1.
        axis = axis / axis.norm(dim=-1, keepdim=True)
        self.axis = torch.where(resample.unsqueeze(-1), axis, self.axis)

    def step(self, substep):
        feet_height = self.asset.data.feet_height
        feet_quat = self.asset.data.body_quat_w[:, self.feet_ids]
        feet_lin_vel = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        forces = (
            self.kp * (self.thres - feet_height) + 
            5. * (0. - feet_lin_vel[:, :, 2])
        ) * self.flag
        self.forces = forces.unsqueeze(-1) * self.axis 
        self.asset._external_force_b[:, self.feet_ids] += quat_rotate_inverse(feet_quat, self.forces)
        self.asset.has_external_wrench = True

    def debug_draw(self):
        feet_pos = self.asset.data.body_link_pos_w[:, self.feet_ids]
        self.env.debug_draw.vector(feet_pos, self.forces / 9.81, color=(0.8, 0.6, 0.6, 1.))


from active_adaptation.envs.mdp.utils.forces import ImpulseForce, ConstantForce
class random_impulse(Randomization):
    def __init__(
        self,
        env,
        prob: float = 0.005,
        body_name: str = None,
        x_range: Tuple[float, float] = (20., 80.),
        y_range: Tuple[float, float] = (20., 80.),
        z_range: Tuple[float, float] = (0., 20.),
        # x_offset_range: Tuple[float, float] = (-0.1, 0.1),
        # y_offset_range: Tuple[float, float] = (-0.1, 0.1),
        # z_offset_range: Tuple[float, float] = (-0.1, 0.1),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.prob = prob
        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0][0]
        else:
            self.body_id = 0 # apply to the root link
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.impulse_force = ImpulseForce.zeros(self.num_envs, device=self.device)
        
    def step(self, substep):
        forces_b = self.asset._external_force_b
        impulse_force = self.impulse_force.get_force(None, None)
        forces_b[:, self.body_id] += quat_rotate_inverse(self.asset.data.root_link_quat_w, impulse_force)
        self.asset.has_external_wrench = True

    def update(self):
        self.impulse_force.time.add_(self.env.step_dt)
        resample = self.impulse_force.expired & (torch.rand(self.num_envs, 1, device=self.device) < self.prob)
        impulse_force = ImpulseForce.sample(self.num_envs, self.device, self.x_range, self.y_range, self.z_range)
        self.impulse_force = impulse_force.where(resample, self.impulse_force)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_id],
            self.impulse_force.get_force(None, None) /  9.81,
            color=(1.0, 0.6, 0.0, 1.0),
            size=3.0,
        )


class constant_force(Randomization):
    def __init__(self, env, force_range, offset_range, body_names = None):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is None:
            self.all_body_ids = torch.tensor([0], device=self.device)
        else:
            self.all_body_ids = torch.tensor(self.asset.find_bodies(body_names)[0], device=self.device)
        
        self.force = ConstantForce.sample(self.num_envs, device=self.device)
        self.force.duration.zero_()
        self.body_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.resample_interval = 50
        self.resample_prob = 0.2

        self.force_range = torch.tensor(force_range, device=self.device)
        self.offset_range = torch.tensor(offset_range, device=self.device)
        
    def step(self, substep):
        arange = torch.arange(self.num_envs, device=self.device)
        quat = self.asset.data.body_quat_w[arange, self.body_id]
        forces_b = quat_rotate_inverse(
            quat.reshape(self.num_envs, 4),
            self.force.get_force()
        )
        self.asset._external_force_b[arange, self.body_id] += forces_b
        self.asset._external_torque_b[arange, self.body_id] += self.force.offset.cross(forces_b, dim=-1)
        self.asset.has_external_wrench = True
    
    def reset(self, env_ids: torch.Tensor):
        self.force.duration.data[env_ids] = 0.
        
    def update(self):
        resample = (self.env.episode_length_buf % self.resample_interval == 0)
        expired = self.force.time > self.force.duration
        resample = resample & expired.squeeze(-1) & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        force = ConstantForce.sample(self.num_envs, self.force_range, self.offset_range, self.device)
        self.force.time.add_(self.env.step_dt)
        self.force = force.where(resample, self.force)
        body_id = self.all_body_ids[torch.randint(0, len(self.all_body_ids), (self.num_envs,), device=self.device)]
        self.body_id = torch.where(resample, body_id, self.body_id)
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[torch.arange(self.num_envs, device=self.device), self.body_id],
            self.force.get_force() /  9.81,
            color=(1.0, 0.6, 0.0, 1.0),
            size=3.0,
        )
        

def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight