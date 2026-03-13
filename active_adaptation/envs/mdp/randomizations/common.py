import torch
import numpy as np
import logging
from typing import Union, Dict, Tuple, Optional

import active_adaptation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils


if active_adaptation.get_backend() == "isaac":
    from isaaclab.actuators import DCMotor, ImplicitActuator
elif active_adaptation.get_backend() == "mjlab":
    import mujoco_warp
    from mjlab.managers.event_manager import RecomputeLevel

    _MJLAB_RECOMPUTE_DERIVED_FIELDS = {
        RecomputeLevel.none: (),
        RecomputeLevel.set_const_fixed: ("body_subtreemass",),
        RecomputeLevel.set_const_0: (
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
        RecomputeLevel.set_const: (
            "body_subtreemass",
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
    }
    _MJLAB_RECOMPUTE_LEVEL_SET_CONST = RecomputeLevel.set_const
else:
    _MJLAB_RECOMPUTE_DERIVED_FIELDS = {}
    _MJLAB_RECOMPUTE_LEVEL_SET_CONST = None

from .base import Randomization

RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


def _mjlab_expand_model_fields(env, *fields: str):
    if env.backend != "mjlab" or len(fields) == 0:
        return
    from mjlab.sim import Simulation
    sim: "Simulation" = env.sim
    if not hasattr(sim, "expand_model_fields") or not hasattr(sim, "expanded_fields"):
        return
    missing = tuple(field for field in fields if field not in sim.expanded_fields)
    if missing:
        sim.expand_model_fields(missing)


def _mjlab_ensure_recompute_fields_expanded(env, level):
    if env.backend != "mjlab" or level is None:
        return
    derived_fields = _MJLAB_RECOMPUTE_DERIVED_FIELDS.get(level, ())
    if derived_fields:
        _mjlab_expand_model_fields(env, *derived_fields)


def _mjlab_recompute_constants(env, level):
    if env.backend != "mjlab":
        return
    from mjlab.sim import Simulation
    sim: "Simulation" = env.sim
    if level is not None:
        _mjlab_ensure_recompute_fields_expanded(env, level)
        if hasattr(sim, "recompute_constants"):
            sim.recompute_constants(level)
            return
    mujoco_warp.set_const(sim.wp_model, sim.wp_data)


def _set_external_wrench(
    asset,
    forces: torch.Tensor,
    torques: torch.Tensor,
    body_ids=None,
):
    """Set external wrench across old/new simulator APIs."""
    if hasattr(asset, "set_external_force_and_torque"):
        kwargs = {}
        if body_ids is not None:
            kwargs["body_ids"] = body_ids
        asset.set_external_force_and_torque(forces, torques, **kwargs)
        return

    if body_ids is None:
        asset._external_force_b[:] = forces
        asset._external_torque_b[:] = torques
    else:
        asset._external_force_b[:, body_ids] = forces
        asset._external_torque_b[:, body_ids] = torques
    if hasattr(asset, "has_external_wrench"):
        asset.has_external_wrench = True


class motor_params(Randomization):
    supported_backends = ("isaac",)
    def __init__(
        self,
        env,
        stiffness_range: Optional[NestedRangeType] = None,
        damping_range: Optional[NestedRangeType] = None,
        armature_range: Optional[NestedRangeType] = None,
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
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
        self.asset = self.env.scene.articulations["robot"]
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
    supported_backends = ("isaac", "mjlab")
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = None,
        dynamic_friction_range = None,
        restitution_range=None,
        solref_time_constant_range=None,
        solref_dampratio_range=None,
        homogeneous: bool=False
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.solref_time_constant_range = solref_time_constant_range
        self.solref_dampratio_range = solref_dampratio_range
        self.homogeneous = homogeneous
        if self.solref_dampratio_range is not None and (
            self.solref_dampratio_range[0] <= 0.0 or self.solref_dampratio_range[1] <= 0.0
        ):
            raise ValueError("solref_dampratio_range must be positive for log-uniform sampling.")

        if self.env.backend == "isaac":
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
            if self.static_friction_range is not None:
                self.static_friction_buckets = torch.linspace(*self.static_friction_range, self.num_buckets)
            if self.dynamic_friction_range is not None:
                self.dynamic_friction_buckets = torch.linspace(*self.dynamic_friction_range, self.num_buckets)
            if self.restitution_range is not None:
                self.restitution_buckets = torch.linspace(*self.restitution_range, self.num_buckets)
        elif self.env.backend == "mjlab":
            _mjlab_expand_model_fields(self.env, "geom_friction", "geom_solref")
            _mjlab_ensure_recompute_fields_expanded(
                self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST
            )
            if self.dynamic_friction_range is not None or self.restitution_range is not None:
                logging.info(
                    "perturb_body_materials(mjlab): dynamic_friction_range/restitution_range are accepted for "
                    "interface compatibility but not applied directly."
                )
            if len(self.body_ids) == 0:
                raise ValueError(
                    "No bodies matched the provided names for material perturbation."
                )

            local_body_ids = torch.as_tensor(
                self.body_ids, device=self.device, dtype=torch.long
            )
            self.global_body_ids = self.asset.indexing.body_ids[local_body_ids]
            selected_body_set = set(self.global_body_ids.cpu().tolist())

            geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
            geom_names = self.asset.geom_names
            selected_geom_local: list[int] = []
            selected_geom_global: list[int] = []
            selected_geom_names: list[str] = []

            cpu_model = self.env.sim.mj_model
            for local_idx, global_idx in enumerate(geom_global_ids):
                body_id = int(cpu_model.geom_bodyid[global_idx])
                if body_id in selected_body_set:
                    selected_geom_local.append(local_idx)
                    selected_geom_global.append(global_idx)
                    selected_geom_names.append(geom_names[local_idx])

            if not selected_geom_global:
                raise ValueError(
                    "No geoms found for the specified bodies when configuring material perturbation."
                )

            self.geom_local_ids = torch.as_tensor(
                selected_geom_local, device=self.device, dtype=torch.long
            )
            self.geom_global_ids = torch.as_tensor(
                selected_geom_global, device=self.device, dtype=torch.long
            )
            self.geom_names = selected_geom_names

    def startup(self):
        if self.env.backend == "isaac":
            logging.info(f"Randomize body materials of {self.body_names} upon startup.")

            materials = self.asset.root_physx_view.get_material_properties().clone()
            if self.homogeneous:
                shape = (self.num_envs, 1)
            else:
                shape = (self.num_envs, len(self.shape_ids))
            if self.static_friction_range is not None:
                materials[:, self.shape_ids, 0] = self.static_friction_buckets[
                    torch.randint(0, self.num_buckets, shape)
                ]
            if self.dynamic_friction_range is not None:
                materials[:, self.shape_ids, 1] = self.dynamic_friction_buckets[
                    torch.randint(0, self.num_buckets, shape)
                ]
            if self.restitution_range is not None:
                materials[:, self.shape_ids, 2] = self.restitution_buckets[
                    torch.randint(0, self.num_buckets, shape)
                ]

            indices = torch.arange(self.asset.num_instances)
            self.asset.root_physx_view.set_material_properties(materials.flatten(), indices)
            self.asset.data.body_materials = materials.to(self.device)
        elif self.env.backend == "mjlab":
            logging.info(f"Randomize body materials of {self.geom_names} upon startup.")

            cpu_model = self.env.sim.mj_model
            # logging.info("perturb_body_materials(mjlab): selected geoms before randomization:")
            # for gid in self.geom_global_ids.tolist():
            #     geom_name = cpu_model.geom(gid).name
            #     body_name = cpu_model.body(cpu_model.geom_bodyid[gid]).name
            #     friction = cpu_model.geom_friction[gid]
            #     priority = cpu_model.geom_priority[gid]
            #     solmix = cpu_model.geom_solmix[gid]
            #     logging.info(
            #         f"  gid={gid} geom={geom_name} body={body_name} "
            #         f"fric={friction} priority={priority} solmix={solmix}"
            #     )

            # logging.info("perturb_body_materials(mjlab): terrain-like geoms before randomization:")
            # for gid in range(cpu_model.ngeom):
            #     geom_name = (cpu_model.geom(gid).name or "")
            #     body_name = (cpu_model.body(cpu_model.geom_bodyid[gid]).name or "")
            #     key = f"{geom_name} {body_name}".lower()
            #     if any(k in key for k in ("ground", "floor", "terrain", "plane")):
            #         friction = cpu_model.geom_friction[gid]
            #         priority = cpu_model.geom_priority[gid]
            #         solmix = cpu_model.geom_solmix[gid]
            #         logging.info(
            #             f"  gid={gid} geom={geom_name} body={body_name} "
            #             f"fric={friction} priority={priority} solmix={solmix}"
            #         )

            num_geoms = self.geom_global_ids.numel()
            sample_cols = 1 if self.homogeneous else num_geoms
            shape = (self.num_envs, sample_cols)

            model = self.env.sim.model
            # model.geom_priority[self.geom_global_ids] = 1
            if self.static_friction_range is not None:
                sf = sample_uniform(shape, *self.static_friction_range, device=self.device)
                if sample_cols == 1:
                    sf = sf.expand(-1, num_geoms)
                model.geom_friction[:, self.geom_global_ids, 0] = sf
                cpu_model.geom_friction[self.geom_global_ids.cpu().numpy()] = (
                    model.geom_friction[0, self.geom_global_ids].to(device="cpu").numpy()
                )
            if self.solref_time_constant_range is not None:
                tc = sample_uniform(shape, *self.solref_time_constant_range, device=self.device)
                if sample_cols == 1:
                    tc = tc.expand(-1, num_geoms)
                model.geom_solref[:, self.geom_global_ids, 0] = tc
                cpu_model.geom_solref[self.geom_global_ids.cpu().numpy(), 0] = (
                    model.geom_solref[0, self.geom_global_ids, 0].to(device="cpu").numpy()
                )
            if self.solref_dampratio_range is not None:
                dr_low, dr_high = self.solref_dampratio_range
                dr = sample_uniform(shape, np.log(dr_low), np.log(dr_high), device=self.device).exp()
                if sample_cols == 1:
                    dr = dr.expand(-1, num_geoms)
                model.geom_solref[:, self.geom_global_ids, 1] = dr
                cpu_model.geom_solref[self.geom_global_ids.cpu().numpy(), 1] = (
                    model.geom_solref[0, self.geom_global_ids, 1].to(device="cpu").numpy()
                )
            _mjlab_recompute_constants(self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST)


class perturb_body_mass(Randomization):
    supported_backends = ("isaac", "mjlab")
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(self.body_ids) == 0:
            raise ValueError("No bodies matched the provided names for mass perturbation.")
        self.mass_ranges = torch.tensor(values, device=self.device, dtype=torch.float32)

        if self.env.backend == "mjlab":
            _mjlab_expand_model_fields(self.env, "body_mass", "body_inertia")
            _mjlab_ensure_recompute_fields_expanded(
                self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST
            )
            self.local_body_ids = torch.as_tensor(
                self.body_ids, device=self.device, dtype=torch.long
            )
            self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
            self._global_body_ids_cpu = self.global_body_ids.to(
                device="cpu", dtype=torch.long
            )
            model = self.env.sim.model
            self._default_body_mass = model.body_mass[:, self.global_body_ids].clone()
            self._default_body_inertia = model.body_inertia[:, self.global_body_ids].clone()

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        if self.env.backend == "isaac":
            masses = self.asset.data.default_mass.clone()
            inertias = self.asset.data.default_inertia.clone()
            scale = uniform(
                self.mass_ranges[:, 0].expand_as(masses[:, self.body_ids]),
                self.mass_ranges[:, 1].expand_as(masses[:, self.body_ids])
            ).cpu()
            masses[:, self.body_ids] *= scale
            inertias[:, self.body_ids] *= scale.unsqueeze(-1)
            indices = torch.arange(self.asset.num_instances)
            self.asset.root_physx_view.set_masses(masses, indices)
            self.asset.root_physx_view.set_inertias(inertias, indices)
            assert torch.allclose(self.asset.root_physx_view.get_masses(), masses)
        elif self.env.backend == "mjlab":
            num_bodies = self.global_body_ids.numel()
            low = self.mass_ranges[:, 0].unsqueeze(0).expand(self.num_envs, num_bodies)
            high = self.mass_ranges[:, 1].unsqueeze(0).expand(self.num_envs, num_bodies)
            scale = uniform(low, high)

            model = self.env.sim.model
            new_mass = self._default_body_mass * scale
            model.body_mass[:, self.global_body_ids] = new_mass
            model.body_inertia[:, self.global_body_ids] = (
                self._default_body_inertia * scale.unsqueeze(-1)
            )
            _mjlab_recompute_constants(self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST)

            # cpu_model = self.env.sim.mj_model
            # cpu_model.body_mass[self._global_body_ids_cpu.numpy()] = (
            #     model.body_mass[0, self.global_body_ids].to(device="cpu").numpy()
            # )
            # cpu_model.body_inertia[self._global_body_ids_cpu.numpy()] = (
            #     model.body_inertia[0, self.global_body_ids].to(device="cpu").numpy()
            # )


class perturb_body_com(Randomization):
    supported_backends = ("isaac", "mjlab")
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]

        self.body_ids, self.body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(self.body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided names for COM perturbation."
            )

        self.pos_ranges = torch.tensor(values, device=self.device, dtype=torch.float32)

        if self.env.backend == "mjlab":
            _mjlab_expand_model_fields(self.env, "body_ipos")
            _mjlab_ensure_recompute_fields_expanded(
                self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST
            )
            self.local_body_ids = torch.as_tensor(
                self.body_ids, device=self.device, dtype=torch.long
            )
            self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
            self._global_body_ids_cpu = self.global_body_ids.to(
                device="cpu", dtype=torch.long
            )
            model = self.env.sim.model
            self._default_body_ipos = model.body_ipos[:, self.global_body_ids].clone()

    def startup(self):
        logging.info(f"Randomize body CoM of {self.body_names} upon startup.")
        if self.env.backend == "isaac":
            coms = self.asset.root_physx_view.get_coms().clone()
            rand_sample = uniform(
                self.pos_ranges[:, 0].unsqueeze(0).unsqueeze(-1).expand_as(coms[:, self.body_ids, :3]),
                self.pos_ranges[:, 1].unsqueeze(0).unsqueeze(-1).expand_as(coms[:, self.body_ids, :3])
            )
            rand_sample[:, :, 0] *= 0.5
            coms[:, self.body_ids, :3] += rand_sample.to('cpu')
            indices = torch.arange(self.asset.num_instances)
            self.asset.root_physx_view.set_coms(coms, indices)
            assert torch.allclose(self.asset.root_physx_view.get_coms(), coms)
        elif self.env.backend == "mjlab":
            num_bodies = self.global_body_ids.numel()
            low = self.pos_ranges[:, 0].unsqueeze(0).unsqueeze(-1).expand(self.num_envs, num_bodies, 3)
            high = self.pos_ranges[:, 1].unsqueeze(0).unsqueeze(-1).expand(self.num_envs, num_bodies, 3)
            offsets = uniform(low, high)

            model = self.env.sim.model
            new_ipos = self._default_body_ipos + offsets
            model.body_ipos[:, self.global_body_ids] = new_ipos
            _mjlab_recompute_constants(self.env, _MJLAB_RECOMPUTE_LEVEL_SET_CONST)

            # cpu_model = self.env.sim.mj_model
            # cpu_model.body_ipos[self._global_body_ids_cpu.numpy()] = (
            #     model.body_ipos[0, self.global_body_ids].to(device="cpu").numpy()
            # )

class perturb_root_vel(Randomization):
    supported_backends = ("isaac", "mjlab")

    def __init__(
        self,
        env,
        min_s: float,
        max_s: float,
        x: Tuple[float, float] = (0.0, 0.0),
        y: Tuple[float, float] = (0.0, 0.0),
        z: Tuple[float, float] = (0.0, 0.0),
        roll: Tuple[float, float] = (0.0, 0.0),
        pitch: Tuple[float, float] = (0.0, 0.0),
        yaw: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]

        self.min_s = float(min_s)
        self.max_s = float(max_s)
        assert 0.0 <= self.min_s <= self.max_s, "Invalid interval for perturbation timing."

        self._range_names = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = (x, y, z, roll, pitch, yaw)
        lows = []
        highs = []
        for name, axis_range in zip(self._range_names, ranges, strict=True):
            if len(axis_range) != 2:
                raise ValueError(f"{name} range must have exactly 2 values, got {axis_range}")
            low = float(axis_range[0])
            high = float(axis_range[1])
            if high < low:
                raise ValueError(f"{name} range must satisfy low <= high, got {axis_range}")
            lows.append(low)
            highs.append(high)
        self.low = torch.tensor(lows, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(highs, dtype=torch.float32, device=self.device)

        self.time_left_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _sample_interval(self, n: int):
        return torch.rand(n, dtype=torch.float32, device=self.device) * (self.max_s - self.min_s) + self.min_s

    def _sample_delta_vel(self, n: int):
        rand = torch.rand((n, 6), dtype=torch.float32, device=self.device)
        return self.low.unsqueeze(0) + (self.high - self.low).unsqueeze(0) * rand

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.time_left_s[env_ids] = self._sample_interval(len(env_ids))

    def update(self):
        self.time_left_s.sub_(self.env.step_dt)
        trigger_ids = torch.nonzero(self.time_left_s <= 1e-6, as_tuple=False).squeeze(-1)
        if trigger_ids.numel() == 0:
            return

        delta_vel = self._sample_delta_vel(trigger_ids.numel())
        root_vel = torch.cat((self.asset.data.root_link_lin_vel_w[trigger_ids], self.asset.data.root_link_ang_vel_w[trigger_ids]), dim=-1)

        self.asset.write_root_link_velocity_to_sim(root_vel + delta_vel, env_ids=trigger_ids)
        # print(f"Applied random root velocity perturbation of {delta_vel} to envs {trigger_ids}.")

        self.time_left_s[trigger_ids] = self._sample_interval(trigger_ids.numel())


class reset_joint_states_uniform(Randomization):
    def __init__(
        self,
        env,
        pos_ranges: Dict[str, tuple],
        vel_ranges: Dict[str, tuple]=None,
        rel: bool=False,
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
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
        self.asset = self.env.scene.articulations["robot"]
        
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


class push_body(Randomization):
    supported_backends = ("isaac", "mujoco")
    def __init__(
        self,
        env,
        body_names,
        force_range = (20, 50),
        min_interval=100,
        decay: float=0.9
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
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

    def pre_step(self, substep):
        _set_external_wrench(
            self.asset,
            self.forces,
            self.torques,
            body_ids=self.body_indices,
        )

    def update(self) -> None:
        t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
        i = torch.rand(self.env.num_envs, len(self.body_indices), 1, device=self.env.device) < 0.02
        i = i & ((t - self.last_push) > self.min_interval)
        self.last_push = torch.where(i, t, self.last_push)

        push_forces = torch.zeros_like(self.forces)
        push_forces[:, :, 0].uniform_(*self.force_range)
        push_forces[:, :, 1].uniform_(*self.force_range)
        self.forces = torch.where(i, push_forces, self.forces * self.decay)
        
    def debug_draw(self):
        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.asset.data.body_link_pos_w[:, self.body_indices],
                self.forces / 9.81,
                color=(1., 0.8, .4, 1.)
            )
        
    
class drag(Randomization):
    def __init__(self, env, body_names, drag_range=(0.0, 0.1)):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
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
        self.asset = self.env.scene.articulations["robot"]
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
        torques = torch.zeros_like(friction_forces)
        _set_external_wrench(
            self.asset,
            friction_forces,
            torques,
            body_ids=self.body_ids,
        )

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
        self.asset = self.env.scene.articulations["robot"]
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
        self.asset = self.env.scene.articulations["robot"]
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
        self.asset = self.env.scene.articulations["robot"]
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
        forces_b = quat_rotate_inverse(feet_quat, self.forces)
        torques_b = torch.zeros_like(forces_b)
        _set_external_wrench(
            self.asset,
            forces_b,
            torques_b,
            body_ids=self.feet_ids,
        )

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
        self.asset = self.env.scene.articulations["robot"]
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
        impulse_force = self.impulse_force.get_force(None, None)
        forces_b = quat_rotate_inverse(self.asset.data.root_link_quat_w, impulse_force)
        torques_b = torch.zeros_like(forces_b)
        _set_external_wrench(
            self.asset,
            forces_b.unsqueeze(1),
            torques_b.unsqueeze(1),
            body_ids=[self.body_id],
        )

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
        self.asset = self.env.scene.articulations["robot"]
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
        torques_b = self.force.offset.cross(forces_b, dim=-1)
        full_forces_b = torch.zeros(
            self.num_envs, self.asset.num_bodies, 3, device=self.device
        )
        full_torques_b = torch.zeros_like(full_forces_b)
        full_forces_b[arange, self.body_id] = forces_b
        full_torques_b[arange, self.body_id] = torques_b
        _set_external_wrench(self.asset, full_forces_b, full_torques_b)
    
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
