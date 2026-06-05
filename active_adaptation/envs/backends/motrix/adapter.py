from __future__ import annotations

import math

import numpy as np
import torch
import motrixsim as mx
from dataclasses import dataclass, field

from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.profiling import ScopedTimer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_adaptation.assets.asset_cfg import EntityCfg

# MotrixSim runs on CPU; keep entity/scene tensors on CPU and copy to numpy at the boundary.
_DEVICE = torch.device("cpu")


def compute_env_origins_grid(
    num_envs: int,
    env_spacing: float,
    device: str | torch.device = _DEVICE,
) -> torch.Tensor:
    """Compute env origins on a centered grid (Isaac Lab compatible layout)."""
    env_origins = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
    if num_envs == 0:
        return env_origins

    num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
    num_cols = math.ceil(num_envs / num_rows)
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=device),
        torch.arange(num_cols, device=device),
        indexing="ij",
    )
    env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
    env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
    return env_origins

PLANE_XML = """
<mujoco model="ground">
  <compiler angle="radian" meshdir="meshes"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <body name="terrain">
      <geom name="terrain_collision" size="0 0 0.05" friction="1 .1 .1" type="plane" material="groundplane"/>
    </body>
  </worldbody>
</mujoco>
"""

@dataclass
class MotrixSceneCfg:
    num_envs: int = 1
    env_spacing: float = 2.0
    entities: dict[str, EntityCfg] = field(default_factory=dict)
    # we leave sensors and terrains to be added later
    sensors: tuple[ContactSensorCfg, ...] = field(default_factory=tuple)


@dataclass
class MotrixEntityData:
    root_link_pose_w: torch.Tensor
    root_link_vel_w: torch.Tensor

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    joint_pos_target: torch.Tensor
    joint_vel_target: torch.Tensor

    body_link_pos_w: torch.Tensor
    body_link_quat_w: torch.Tensor
    body_vel_w: torch.Tensor

    default_root_state: torch.Tensor
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor

    @property
    def heading_w(self) -> torch.Tensor:
        forward_w = quat_rotate(
            self.root_link_quat_w,
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])
    
    @property
    def projected_gravity_b(self) -> torch.Tensor:
        return quat_rotate(
            self.root_link_quat_w,
            torch.tensor([[0.0, 0.0, -1.0]]),
        )

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self.root_link_pose_w[:, :3]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.root_link_pos_w

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        # MotrixSim pose uses xyzw; mjlab/Isaac use wxyz in many MDP terms.
        q = self.root_link_pose_w[:, 3:7]
        return q[:, [3, 0, 1, 2]]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.root_link_quat_w

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self.root_link_vel_w[:, :3]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.root_link_lin_vel_w

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self.root_link_vel_w[:, 3:]
    
    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        # TODO: use true com velocity
        return self.root_link_vel_w[:, :3]
    
    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        return quat_rotate_inverse(
            self.root_link_quat_w,
            self.root_com_lin_vel_w,
        )

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        return self.root_link_vel_w[:, 3:] # same as root_link_ang_vel_w
    
    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        return quat_rotate_inverse(
            self.root_link_quat_w,
            self.root_com_ang_vel_w,
        )

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.root_link_ang_vel_w
    
    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self.body_vel_w[:, :, :3]
    
    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self.body_vel_w[:, :, 3:]


REDUCE_MAP = {
    "mindist": mx.msd.ContactSensorReduce.MinDist,
    "maxforce": mx.msd.ContactSensorReduce.MaxForce,
    "netforce": mx.msd.ContactSensorReduce.NetForce,
}


class MotrixScene:
    def __init__(self, cfg: MotrixSceneCfg):
        self.cfg = cfg
        self._entities: dict[str, MotrixEntity] = {}
        self._sensors: dict[str, MotrixContactSensor] = {}
        self._msd_world = mx.msd.from_str(PLANE_XML)        

        self._add_entities()
        self._add_sensors()
        # TODO: add terrain

        self.mx_model: mx.SceneModel = self._msd_world.build()
        # TODO: make these configurable
        self.mx_model.options.max_iterations = 6
        self.mx_model.options.timestep = 0.01

        self.mx_data = mx.SceneData(self.mx_model, batch=(self.cfg.num_envs,))
        self.mx_model.step(self.mx_data)

        self._env_origins = compute_env_origins_grid(
            self.cfg.num_envs,
            self.cfg.env_spacing,
        )
        for ent in self._entities.values():
            ent._initialize(self.mx_model, self.mx_data)
        for sensor in self._sensors.values():
            sensor._initialize(self.mx_model, self.mx_data)
    
    def _add_entities(self):
        for name, cfg in self.cfg.entities.items():
            ent = MotrixEntity(cfg)
            prefix = name + "/"
            ent._prefix = prefix
            self._msd_world.attach(
                ent.msd,
                other_translation=ent.cfg.init_state.pos,
                other_rotation=wxyz2xyzw(torch.tensor(ent.cfg.init_state.rot)).tolist(),
                other_prefix=prefix,
            )
            self._entities[name] = ent
    
    def _add_sensors(self):
        for sensor_cfg in self.cfg.sensors:
            sensor = MotrixContactSensor(sensor_cfg)
            sensor.edit_spec(self._msd_world, self._entities)
            self._sensors[sensor_cfg.name] = sensor

    def update(self, dt: float):
        with ScopedTimer("update_entities"):
            for ent in self._entities.values():
                ent.update(dt)
        with ScopedTimer("update_sensors"):
            for sensor in self._sensors.values():
                sensor.update(dt)
    
    @property
    def entities(self) -> dict[str, MotrixEntity]:
        return dict(self._entities)
    
    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs
    
    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env spawn offsets, shape ``(num_envs, 3)``."""
        return self._env_origins

    @property
    def render_offsets(self) -> list[list[float]]:
        """Render-only offsets for ``RenderApp.launch(..., render_offset=...)``."""
        return self._env_origins.cpu().tolist()
    
    @property
    def articulations(self) -> dict[str, MotrixEntity]:
        return dict(self._entities)

    @property
    def sensors(self) -> dict:
        return dict(self._sensors)

    @property
    def device(self) -> str:
        return str(_DEVICE)

    def __getitem__(self, key: str) -> MotrixEntity:
        return self._entities[key]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset selected envs to entity defaults."""
        for ent in self._entities.values():
            ent.reset(env_ids)

    def write_data_to_sim(self) -> None:
        for ent in self._entities.values():
            ent.write_data_to_sim()

    def zero_external_wrenches(self) -> None:
        pass

    def get_spawn_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.env_origins[env_ids]

    @property
    def ground_mesh(self):
        return None # TODO: implement ground mesh


class MotrixSim:
    """Minimal physics runtime facade over ``MotrixScene``."""

    device = "cpu"

    def __init__(self, scene: MotrixScene, headless: bool = True):
        self.scene = scene
        self._render_app = None
        self._render_ctx = None
        if not headless:
            from motrixsim.render import RenderApp

            self._render_ctx = RenderApp()
            self._render_app = self._render_ctx.__enter__()
            self._render_app.launch(
                scene.mx_model,
                batch=scene.num_envs,
                render_offset=scene.render_offsets,
            )

    def get_physics_dt(self) -> float:
        return float(self.scene.mx_model.options.timestep)

    def has_gui(self) -> bool:
        return self._render_app is not None

    def step(self, render: bool = False) -> None:
        self.scene.mx_model.step(self.scene.mx_data)

    def render(self) -> None:
        if self._render_app is not None:
            self._render_app.sync(self.scene.mx_data)

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        pass

    def close(self) -> None:
        if self._render_ctx is not None:
            self._render_ctx.__exit__(None, None, None)
            self._render_ctx = None
            self._render_app = None


class MotrixEntity:
    def __init__(self, cfg: EntityCfg):
        self.cfg = cfg
        # Reuse mjlab's Entity pipeline to inject actuators/collisions into MJCF,
        # then load the exported file via MSD. See motrix/mjcf.py.
        if cfg.motrix_mjcf_path_fn is None:
            raise ValueError(
                "EntityCfg.motrix_mjcf_path_fn is required for MotrixEntity. "
                "Set it in the asset's make_mjlab_cfg(motrix=True) factory."
            )
        # TODO: implement `edit_spec` to programatically add actuators
        self.mjcf_path = cfg.motrix_mjcf_path_fn(cfg)
        self.msd = mx.msd.from_file(self.mjcf_path)
        self._joint_names = [
            a.name for a in self.msd.actuators
        ]
        self._body_names: list[str] = []
        self._geom_names: list[str] = []
        
        links: list[mx.msd.Link] = [self.msd.hierarchy.bodies[0].link]
        while len(links):
            link = links.pop(0)
            self._body_names.append(link.name)
            self._geom_names.extend(geom.name for geom in link.geoms)
            links.extend(link.children)

        self._data = None
        self._body = None
        self._base_link = None
        self._floatingbase = None
        self._actuator_ctrl: torch.Tensor | None = None
        self._mx_model = None
        self._mx_data = None
        self._prefix = None

    @staticmethod
    def _as_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float32, device=_DEVICE)
        return torch.as_tensor(value, dtype=torch.float32, device=_DEVICE)

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.cpu().numpy().astype(np.float32, copy=False)    

    def _view_data(self, env_ids: torch.Tensor | slice | None) -> mx.SceneData:
        if env_ids is None:
            return self._mx_data
        elif isinstance(env_ids, torch.Tensor):
            env_ids = mx.DisjointIndices(env_ids.numpy())
        return self._mx_data[env_ids]

    @staticmethod
    def _assign_env_columns(
        tensor: torch.Tensor,
        values: torch.Tensor,
        env_sel: torch.Tensor | slice,
        column_ids: torch.Tensor | slice,
    ) -> None:
        """Assign ``values`` into ``tensor[envs, columns]`` without chained indexing."""
        if isinstance(env_sel, torch.Tensor):
            tensor[env_sel[:, None], column_ids] = values
        else:
            tensor[env_sel, column_ids] = values

    def _sync_kinematics(self) -> None:
        self._mx_model.forward_kinematic(self._mx_data)

    def _read_root_link_vel_w(self) -> torch.Tensor:
        # Body has no velocity API; use the root link (see link.py / geom.py examples).
        lin_vel = torch.from_numpy(self._base_link.get_linear_velocity(self._mx_data))
        ang_vel = torch.from_numpy(self._base_link.get_angular_velocity(self._mx_data))
        return torch.cat([lin_vel, ang_vel], dim=-1,)

    def _initialize(
        self,
        mx_model: mx.SceneModel,
        mx_data: mx.SceneData,
    ):
        self._mx_model = mx_model
        self._mx_data = mx_data
        base: mx.msd.Body = self.msd.hierarchy.bodies[0]
        self.base_link_name = base.link.name
        self.base_link_path = f"{self._prefix}{self.base_link_name}"
        self._body = mx_model.get_body(self.base_link_path)
        if self._body is None:
            raise ValueError(f"Body {self.base_link_path} not found in model")
        self._base_link = self._body.base_link
        self._floatingbase = self._body.floatingbase
        if self._floatingbase is None:
            raise ValueError(f"Body {self.base_link_path} has no floating base")

        self._body.base_link.joint_indices
        from mjlab.utils.string import resolve_expr
        joint_pos = self._as_tensor(self._body.get_joint_dof_pos(mx_data))
        joint_vel = self._as_tensor(self._body.get_joint_dof_vel(mx_data))
        num_envs = joint_pos.shape[0]
        num_actuators = self._body.num_actuators
        self._actuator_ctrl = torch.zeros(
            num_envs, num_actuators, dtype=torch.float32, device=_DEVICE
        )

        default_joint_pos = torch.tensor(
            resolve_expr(self.cfg.init_state.joint_pos, self._joint_names, 0.0),
            dtype=torch.float32,
            device=_DEVICE,
        ).unsqueeze(0).expand(num_envs, -1)
        default_joint_vel = torch.tensor(
            resolve_expr(self.cfg.init_state.joint_vel, self._joint_names, 0.0),
            dtype=torch.float32,
            device=_DEVICE,
        ).unsqueeze(0).expand(num_envs, -1)

        pose = self._as_tensor(self._body.get_pose(mx_data))
        root_link_vel_w = self._read_root_link_vel_w()
        root_state = torch.cat([pose, root_link_vel_w], dim=-1)
        
        self._link_indices = torch.tensor([
            self._mx_model.get_link_index(f"{self._prefix}{body_name}")
            for body_name in self._body_names
        ])
        body_pose_w = self._mx_model.get_link_poses(mx_data)[:, self._link_indices]
        body_pos_w = body_pose_w[:, :, :3]
        body_quat_w = xyzw2wxyz(body_pose_w[:, :, 3:7])
        body_vel_w = self._mx_model.get_link_linear_velocities(mx_data)[:, self._link_indices]

        self._data = MotrixEntityData(
            root_link_pose_w=pose,
            root_link_vel_w=root_link_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_acc=torch.zeros_like(joint_vel),
            joint_pos_target=default_joint_pos.clone(),
            joint_vel_target=default_joint_vel.clone(),
            body_link_pos_w=torch.from_numpy(body_pos_w),
            body_link_quat_w=torch.from_numpy(body_quat_w),
            body_vel_w=body_vel_w,
            default_root_state=root_state,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
        )
    
    def update(self, dt: float):
        self._data.root_link_pose_w = self._as_tensor(self._body.get_pose(self._mx_data))
        self._data.root_link_vel_w = self._read_root_link_vel_w()

        joint_pos = torch.from_numpy(self._body.get_joint_dof_pos(self._mx_data))
        joint_vel = torch.from_numpy(self._body.get_joint_dof_vel(self._mx_data))
        joint_acc = (joint_vel - self._data.joint_vel) / dt

        body_pose_w = self._mx_model.get_link_poses(self._mx_data)[:, self._link_indices]
        body_pos_w = torch.from_numpy(body_pose_w[:, :, :3])
        body_quat_w = torch.from_numpy(xyzw2wxyz(body_pose_w[:, :, 3:7]))
        body_vel_w = self._mx_model.get_link_linear_velocities(self._mx_data)[:, self._link_indices]

        self._data.joint_pos = joint_pos
        self._data.joint_vel = joint_vel
        self._data.joint_acc = joint_acc
        self._data.body_link_pos_w = body_pos_w
        self._data.body_link_quat_w = body_quat_w
        self._data.body_vel_w = torch.from_numpy(body_vel_w)

    @property
    def data(self) -> MotrixEntityData:
        return self._data

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names) # copy the list to avoid modifying the original

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names) # copy the list to avoid modifying the original
    
    @property
    def geom_names(self) -> list[str]:
        return list(self._geom_names) # copy the list to avoid modifying the original

    def find_joints(
        self,
        name_keys: str | list[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        from mjlab.utils.lab_api.string import resolve_matching_names

        if joint_subset is None:
            joint_subset = self._joint_names
        return resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_bodies(
        self,
        name_keys: str | list[str],
        body_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        from mjlab.utils.lab_api.string import resolve_matching_names
        if body_subset is None:
            body_subset = self._body_names
        return resolve_matching_names(name_keys, body_subset, preserve_order)

    def find_geoms(
        self,
        name_keys: str | list[str],
        geom_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        from mjlab.utils.lab_api.string import resolve_matching_names
        if geom_subset is None:
            geom_subset = self._geom_names
        return resolve_matching_names(name_keys, geom_subset, preserve_order)

    def set_joint_position_target(
        self,
        position: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        env_sel = slice(None) if env_ids is None else env_ids
        position = self._as_tensor(position)
        if joint_ids is None or joint_ids == slice(None):
            self._data.joint_pos_target[env_sel] = position
        else:
            self._assign_env_columns(
                self._data.joint_pos_target, position, env_sel, joint_ids
            )

    def set_joint_velocity_target(
        self,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        env_sel = slice(None) if env_ids is None else env_ids
        velocity = self._as_tensor(velocity)
        if joint_ids is None or joint_ids == slice(None):
            self._data.joint_vel_target[env_sel] = velocity
        else:
            self._assign_env_columns(
                self._data.joint_vel_target, velocity, env_sel, joint_ids
            )

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Write root state. Input layout matches mjlab: pos(3), quat wxyz(4), lin_vel(3), ang_vel(3)."""
        assert root_state.shape[-1] == 13
        env_sel = slice(None) if env_ids is None else env_ids
        root_state = self._as_tensor(root_state)
        quat_xyzw = wxyz2xyzw(root_state[:, 3:7])
        pose = torch.cat([root_state[:, :3], quat_xyzw], dim=-1)
        self._data.root_link_pose_w[env_sel] = pose
        self._data.root_link_vel_w[env_sel] = root_state[:, 7:13]

        data_view = self._view_data(env_ids)
        translation = np.ascontiguousarray(root_state[:, :3])
        quat_xyzw = np.ascontiguousarray(quat_xyzw)
        lin_vel = np.ascontiguousarray(root_state[:, 7:10])
        ang_vel = np.ascontiguousarray(root_state[:, 10:13])

        self._floatingbase.set_translation(data_view, translation)
        self._floatingbase.set_rotation(data_view, quat_xyzw)
        self._floatingbase.set_global_linear_velocity(data_view, lin_vel)
        self._floatingbase.set_global_angular_velocity(data_view, ang_vel)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        env_sel = slice(None) if env_ids is None else env_ids
        position = self._as_tensor(position)
        velocity = self._as_tensor(velocity)

        if joint_ids is None or joint_ids == slice(None):
            self._data.joint_pos[env_sel] = position
            self._data.joint_vel[env_sel] = velocity
            pos = position
            vel = velocity
        else:
            self._assign_env_columns(self._data.joint_pos, position, env_sel, joint_ids)
            self._assign_env_columns(self._data.joint_vel, velocity, env_sel, joint_ids)
            pos = self._data.joint_pos[env_sel]
            vel = self._data.joint_vel[env_sel]

        data_view = self._view_data(env_ids)
        # Third arg excludes the free joint; root is written separately.
        self._body.set_dof_pos(data_view, self._to_numpy(pos), False)
        self._body.set_dof_vel(data_view, self._to_numpy(vel), False)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_sel = slice(None) if env_ids is None else env_ids
        self._data.joint_pos[env_sel] = self._data.default_joint_pos[env_sel]
        self._data.joint_vel[env_sel] = self._data.default_joint_vel[env_sel]
        self._data.joint_pos_target[env_sel] = self._data.default_joint_pos[env_sel]
        self._data.joint_vel_target[env_sel] = self._data.default_joint_vel[env_sel]
        self.write_joint_state_to_sim(
            self._data.joint_pos[env_sel],
            self._data.joint_vel[env_sel],
            env_ids=env_ids,
        )

    def write_data_to_sim(self) -> None:
        """Push joint position targets to MotrixSim position actuators."""
        self._actuator_ctrl.copy_(self._data.joint_pos_target)
        self._body.set_actuator_ctrls(
            self._mx_data,
            self._to_numpy(self._data.joint_pos_target),
        )

    def write_ctrl_to_sim(
        self,
        ctrl: torch.Tensor,
        ctrl_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        raise NotImplementedError("write_ctrl_to_sim is not implemented for MotrixEntity")
        env_sel = slice(None) if env_ids is None else env_ids
        ctrl = self._as_tensor(ctrl)
        if ctrl_ids is None or ctrl_ids == slice(None):
            self._actuator_ctrl[env_sel] = ctrl
        else:
            self._assign_env_columns(self._actuator_ctrl, ctrl, env_sel, ctrl_ids)

        data_view = self._view_data(env_ids)
        self._body.set_actuator_ctrls(
            data_view,
            self._to_numpy(self._actuator_ctrl[env_sel]),
        )


import re
from dataclasses import dataclass
from typing import Literal

from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg, ContactData


class MotrixContactSensor:
    """Add Motrix MSD contact sensors from an mjlab :class:`ContactSensorCfg`.
    
    We reuse the mjlab's ContactSensorCfg and ContactData classes for convenience.
    """

    _REPORT_FIELDS = frozenset(
        {"found", "force", "torque", "dist", "pos", "normal", "tangent"}
    )
    _FIELD_ORDER = ("found", "force", "torque", "dist", "pos", "normal", "tangent")
    _FIELD_DIMS = {
        "found": 1,
        "force": 3,
        "torque": 3,
        "dist": 1,
        "pos": 3,
        "normal": 3,
        "tangent": 3,
    }

    def __init__(self, cfg: ContactSensorCfg) -> None:
        self.cfg = cfg
        self._primary_names: list[str] = []
        self._sensor_names: list[str] = []
        self._sensor_stride = 0
        self._mx_model: mx.SceneModel | None = None
        self._mx_data: mx.SceneData | None = None
        self._cached_data: ContactData | None = None
        self._cache_valid = False
        self._air_time_state: _MotrixAirTimeState | None = None
        self._history_state: dict[str, torch.Tensor] | None = None

        if cfg.global_frame and cfg.reduce != "netforce":
            if "normal" not in cfg.fields or "tangent" not in cfg.fields:
                raise ValueError(
                    f"Sensor '{cfg.name}': global_frame=True requires 'normal' and "
                    "'tangent' in fields"
                )
        if cfg.track_air_time and "found" not in cfg.fields:
            raise ValueError(
                f"Sensor '{cfg.name}': track_air_time=True requires 'found' in fields"
            )
    
    def find_bodies(
        self,
        name_keys: str | list[str],
        body_subset: list[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        from mjlab.utils.lab_api.string import resolve_matching_names
        if body_subset is None:
            body_subset = self.primary_names
        return resolve_matching_names(name_keys, body_subset, preserve_order)

    def edit_spec(
        self,
        msd_scene: mx.msd.World,
        entities: dict[str, MotrixEntity] | None = None,
    ) -> None:
        """Expand patterns and register one MSD contact sensor per primary."""
        self._primary_names.clear()
        self._sensor_names.clear()

        primary_names = self._resolve_primary_names(entities, self.cfg.primary)
        if self.cfg.secondary is None or self.cfg.secondary_policy == "any":
            secondary_name = None
        else:
            secondary_name = self._resolve_single_secondary(
                entities,
                self.cfg.secondary,
                self.cfg.secondary_policy,
            )

        if secondary_name is None:
            raise ValueError(
                f"Sensor '{self.cfg.name}': Motrix contact sensors require an explicit "
                "secondary target (secondary_policy='any' is not supported)."
            )

        for prim in primary_names:
            sensor_name = f"{self.cfg.name}_{prim}"
            self._add_contact_sensor_to_spec(
                msd_scene,
                sensor_name,
                self._sim_name(prim, self.cfg.primary.entity, entities),
                self._sim_name(secondary_name, self.cfg.secondary.entity, entities),
                self.cfg.fields,
            )
            self._primary_names.append(prim)
    
    @property
    def primary_names(self) -> list[str]:
        return list(self._primary_names)

    @property
    def data(self) -> ContactData:
        if not self._cache_valid:
            self._cached_data = self._compute_data()
            self._cache_valid = True
        assert self._cached_data is not None
        return self._cached_data

    def _initialize(self, mx_model: mx.SceneModel, mx_data: mx.SceneData) -> None:
        if not self._sensor_names:
            raise RuntimeError(
                f"There was an error initializing contact sensor '{self.cfg.name}'"
            )

        self._mx_model = mx_model
        self._mx_data = mx_data
        self._sensor_stride = sum(
            self._FIELD_DIMS[field]
            for field in self._FIELD_ORDER
            if field in self.cfg.fields
        )
        self._cache_valid = False

        n_primary = len(self._primary_names)
        n_envs = int(mx_data.shape[0])

        if self.cfg.track_air_time:
            self._air_time_state = _MotrixAirTimeState(
                current_air_time=torch.zeros((n_envs, n_primary), device=_DEVICE),
                last_air_time=torch.zeros((n_envs, n_primary), device=_DEVICE),
                current_contact_time=torch.zeros((n_envs, n_primary), device=_DEVICE),
                last_contact_time=torch.zeros((n_envs, n_primary), device=_DEVICE),
            )

        if self.cfg.history_length > 0:
            n_contacts = n_primary * self.cfg.num_slots
            h = self.cfg.history_length
            self._history_state = {}
            if "force" in self.cfg.fields:
                self._history_state["force"] = torch.zeros(
                    (n_envs, n_contacts, h, 3), device=_DEVICE
                )
            if "torque" in self.cfg.fields:
                self._history_state["torque"] = torch.zeros(
                    (n_envs, n_contacts, h, 3), device=_DEVICE
                )
            if "dist" in self.cfg.fields:
                self._history_state["dist"] = torch.zeros(
                    (n_envs, n_contacts, h), device=_DEVICE
                )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_sel = slice(None) if env_ids is None else env_ids
        self._cache_valid = False

        if self._air_time_state is not None:
            self._air_time_state.current_air_time[env_sel] = 0.0
            self._air_time_state.last_air_time[env_sel] = 0.0
            self._air_time_state.current_contact_time[env_sel] = 0.0
            self._air_time_state.last_contact_time[env_sel] = 0.0

        if self._history_state is not None:
            for buf in self._history_state.values():
                buf[env_sel] = 0.0

    def update(self, dt: float) -> None:
        if self._mx_model is None or self._mx_data is None:
            raise RuntimeError(f"Sensor '{self.cfg.name}' not initialized")

        raw = self._mx_model.get_sensor_values(self._sensor_names, self._mx_data)
        contact_data = self._parse_sensor_values(raw)

        if self.cfg.global_frame and self.cfg.reduce != "netforce":
            contact_data = self._transform_to_global_frame(contact_data)

        if self._air_time_state is not None and contact_data.found is not None:
            elapsed_time = torch.full(
                (contact_data.found.shape[0],),
                float(dt),
                device=_DEVICE,
            ).unsqueeze(-1)
            found = contact_data.found
            if self.cfg.num_slots > 1:
                found = found.view(found.shape[0], -1, self.cfg.num_slots).any(dim=-1)
            is_contact = found > 0

            state = self._air_time_state
            is_first_contact = (state.current_air_time > 0) & is_contact
            is_first_detached = (state.current_contact_time > 0) & ~is_contact

            state.last_air_time[:] = torch.where(
                is_first_contact,
                state.current_air_time + elapsed_time,
                state.last_air_time,
            )
            state.current_air_time[:] = torch.where(
                ~is_contact,
                state.current_air_time + elapsed_time,
                torch.zeros_like(state.current_air_time),
            )
            state.last_contact_time[:] = torch.where(
                is_first_detached,
                state.current_contact_time + elapsed_time,
                state.last_contact_time,
            )
            state.current_contact_time[:] = torch.where(
                is_contact,
                state.current_contact_time + elapsed_time,
                torch.zeros_like(state.current_contact_time),
            )

            contact_data.current_air_time = state.current_air_time
            contact_data.last_air_time = state.last_air_time
            contact_data.current_contact_time = state.current_contact_time
            contact_data.last_contact_time = state.last_contact_time

        if self._history_state is not None:
            if "force" in self._history_state and contact_data.force is not None:
                self._history_state["force"] = self._history_state["force"].roll(1, dims=2)
                self._history_state["force"][:, :, 0, :] = contact_data.force
            if "torque" in self._history_state and contact_data.torque is not None:
                self._history_state["torque"] = self._history_state["torque"].roll(1, dims=2)
                self._history_state["torque"][:, :, 0, :] = contact_data.torque
            if "dist" in self._history_state and contact_data.dist is not None:
                self._history_state["dist"] = self._history_state["dist"].roll(1, dims=2)
                self._history_state["dist"][:, :, 0] = contact_data.dist

            contact_data.force_history = self._history_state.get("force")
            contact_data.torque_history = self._history_state.get("torque")
            contact_data.dist_history = self._history_state.get("dist")

        self._cached_data = contact_data
        self._cache_valid = True
    
    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        if self._air_time_state is None:
            raise RuntimeError(f"Sensor '{self.cfg.name}' must have track_air_time=True to use compute_first_contact")
        is_in_contact = self._air_time_state.current_contact_time > 0.0
        within_dt = self._air_time_state.current_contact_time < (dt + abs_tol)
        return is_in_contact & within_dt
    
    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        if self._air_time_state is None:
            raise RuntimeError(f"Sensor '{self.cfg.name}' must have track_air_time=True to use compute_first_air")
        is_in_air = self._air_time_state.current_air_time > 0.0
        within_dt = self._air_time_state.current_air_time < (dt + abs_tol)
        return is_in_air & within_dt

    def _compute_data(self) -> ContactData:
        if self._mx_model is None or self._mx_data is None:
            raise RuntimeError(f"Sensor '{self.cfg.name}' not initialized")
        raw = self._mx_model.get_sensor_values(self._sensor_names, self._mx_data)
        contact_data = self._parse_sensor_values(raw)
        if self.cfg.global_frame and self.cfg.reduce != "netforce":
            contact_data = self._transform_to_global_frame(contact_data)
        if self._air_time_state is not None:
            contact_data.current_air_time = self._air_time_state.current_air_time
            contact_data.last_air_time = self._air_time_state.last_air_time
            contact_data.current_contact_time = self._air_time_state.current_contact_time
            contact_data.last_contact_time = self._air_time_state.last_contact_time
        if self._history_state is not None:
            contact_data.force_history = self._history_state.get("force")
            contact_data.torque_history = self._history_state.get("torque")
            contact_data.dist_history = self._history_state.get("dist")
        return contact_data

    def _parse_sensor_values(self, raw) -> ContactData:
        n_primary = len(self._primary_names)
        if n_primary == 0:
            raise RuntimeError(f"Sensor '{self.cfg.name}' has no primaries")

        values = torch.as_tensor(raw, dtype=torch.float32, device=_DEVICE)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        per_primary = values.view(values.shape[0], n_primary, self._sensor_stride)

        field_offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for field in self._FIELD_ORDER:
            if field not in self.cfg.fields:
                continue
            dim = self._FIELD_DIMS[field]
            field_offsets[field] = (cursor, cursor + dim)
            cursor += dim

        out = ContactData()
        for field in self.cfg.fields:
            start, end = field_offsets[field]
            chunk = per_primary[..., start:end]
            if chunk.shape[-1] == 1:
                chunk = chunk.squeeze(-1)
            setattr(out, field, chunk)
        return out

    def _transform_to_global_frame(self, data: ContactData) -> ContactData:
        assert data.normal is not None and data.tangent is not None

        normal = data.normal
        tangent = data.tangent
        tangent2 = torch.cross(normal, tangent, dim=-1)
        rot = torch.stack([tangent, tangent2, normal], dim=-1)

        has_contact = torch.norm(normal, dim=-1, keepdim=True) > 1e-8

        if data.force is not None:
            force_global = torch.einsum("...ij,...j->...i", rot, data.force)
            data.force = torch.where(has_contact, force_global, data.force)

        if data.torque is not None:
            torque_global = torch.einsum("...ij,...j->...i", rot, data.torque)
            data.torque = torch.where(has_contact, torque_global, data.torque)

        return data

    @staticmethod
    def _entity_prefix(
        entity_key: str | None,
        entities: dict[str, MotrixEntity],
    ) -> str:
        if entity_key in (None, ""):
            return ""
        if entity_key in entities:
            prefix = entities[entity_key]._prefix
            return prefix if prefix is not None else f"{entity_key}/"
        return f"{entity_key}/"

    @classmethod
    def _sim_name(
        cls,
        name: str,
        entity_key: str | None,
        entities: dict[str, MotrixEntity],
    ) -> str:
        prefix = cls._entity_prefix(entity_key, entities)
        if not prefix:
            return name
        return name if name.startswith(prefix) else f"{prefix}{name}"

    @classmethod
    def _apply_excludes(cls, names: list[str], excludes: tuple[str, ...]) -> list[str]:
        if not excludes:
            return names
        exclude_patterns: list[re.Pattern[str]] = []
        exclude_exact: set[str] = set()
        for exc in excludes:
            if any(c in exc for c in r".*+?[]{}()\|^$"):
                exclude_patterns.append(re.compile(exc))
            else:
                exclude_exact.add(exc)
        if exclude_exact:
            names = [n for n in names if n not in exclude_exact]
        if exclude_patterns:
            names = [n for n in names if not any(rx.search(n) for rx in exclude_patterns)]
        return names

    def _resolve_primary_names(
        self,
        entities: dict[str, MotrixEntity],
        match: ContactMatch,
    ) -> list[str]:

        if match.entity in (None, ""):
            patterns = [match.pattern] if isinstance(match.pattern, str) else list(match.pattern)
            return patterns

        if match.entity not in entities and not self._entity_prefix(match.entity, entities):
            raise ValueError(
                f"Primary entity '{match.entity}' not found. "
                f"Available: {list(entities.keys())}"
            )
        ent = entities[match.entity]

        patterns = [match.pattern] if isinstance(match.pattern, str) else list(match.pattern)

        if match.mode == "geom":
            _, names = ent.find_geoms(patterns)
        elif match.mode in ("body", "subtree"):
            _, names = ent.find_bodies(patterns)
        else:
            raise ValueError("Primary mode must be one of {'geom','body','subtree'}")

        names = self._apply_excludes(names, match.exclude)

        if not names:
            raise ValueError(
                f"Primary pattern '{match.pattern}' (after excludes) matched "
                f"no names in '{match.entity}'"
            )
        return names

    def _resolve_single_secondary(
        self,
        entities: dict[str, MotrixEntity],
        match: ContactMatch,
        policy: Literal["first", "any", "error"],
    ) -> str | None:
        if policy == "any":
            return None

        if isinstance(match.pattern, tuple):
            raise ValueError(
                "Secondary must specify a single name (string). "
                "Use a single exact name or a regex that resolves to one name, "
                "or set secondary_policy='any' if you want no filter."
            )

        if match.entity in (None, ""):
            if match.mode not in {"geom", "body", "subtree"}:
                raise ValueError("Secondary mode must be one of {'geom','body','subtree'}")
            return match.pattern

        if match.entity not in entities and not self._entity_prefix(match.entity, entities):
            raise ValueError(
                f"Secondary entity '{match.entity}' not found. "
                f"Available: {list(entities.keys())}"
            )
        ent = entities[match.entity]

        if match.mode == "subtree":
            return match.pattern

        if match.mode == "geom":
            _, names = ent.find_geoms(match.pattern)
        elif match.mode == "body":
            _, names = ent.find_bodies(match.pattern)
        else:
            raise ValueError("Secondary mode must be one of {'geom','body','subtree'}")

        if not names:
            raise ValueError(
                f"Secondary pattern '{match.pattern}' matched nothing in '{match.entity}'"
            )

        if len(names) == 1 or policy == "first":
            return names[0]

        raise ValueError(
            f"Secondary pattern '{match.pattern}' matched multiple: {names}. "
            "Be explicit or set secondary_policy='first' or 'any'."
        )

    def _add_contact_sensor_to_spec(
        self,
        msd_world: mx.msd.World,
        sensor_name: str,
        primary_name: str,
        secondary_name: str,
        fields: tuple[str, ...],
    ) -> None:
        if self.cfg.secondary is not None and (
            self.cfg.primary.mode != self.cfg.secondary.mode
        ):
            raise ValueError("Primary and secondary modes must be the same")
        if self.cfg.num_slots != 1:
            raise ValueError("num_slots must be 1 for motrix")

        match_fn = {
            "geom": mx.msd.ContactMatch.geom_pair,
            "body": mx.msd.ContactMatch.link_pair,
            "subtree": mx.msd.ContactMatch.subtree_pair,
        }
        sensor = mx.msd.ContactSensor()
        sensor.name = sensor_name
        sensor.match_ = match_fn[self.cfg.primary.mode](primary_name, secondary_name)
        sensor.report = mx.msd.ContactSensorReport()

        requested = set(fields)
        unrecognized = requested - self._REPORT_FIELDS
        if unrecognized:
            raise ValueError("Unrecognized fields: " + ", ".join(sorted(unrecognized)))

        for field in self._REPORT_FIELDS:
            setattr(sensor.report, field, field in requested)

        sensor.reduce = REDUCE_MAP[self.cfg.reduce]
        sensor.max_num = self.cfg.num_slots
        msd_world.sensors.contact.append(sensor)
        self._sensor_names.append(sensor_name)

        if self.cfg.debug:
            print(
                "Adding Motrix contact sensor\n"
                f"  name     : {sensor_name}\n"
                f"  primary  : {primary_name}\n"
                f"  secondary: {secondary_name}\n"
                f"  fields   : {', '.join(fields)}\n"
                f"  reduce   : {self.cfg.reduce}  num_slots={self.cfg.num_slots}"
            )


@dataclass
class _MotrixAirTimeState:
    current_air_time: torch.Tensor
    last_air_time: torch.Tensor
    current_contact_time: torch.Tensor
    last_contact_time: torch.Tensor


def wxyz2xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz[..., [1, 2, 3, 0]]


def xyzw2wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[..., [3, 0, 1, 2]]
