from __future__ import annotations

import math

import numpy as np
import torch
import motrixsim as mx
from dataclasses import dataclass, field

from active_adaptation.assets.asset_cfg import EntityCfg
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

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

plane_xml = """
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
    <geom name="floor" size="0 0 0.05" friction="1 .1 .1" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

@dataclass
class MotrixSceneCfg:
    num_envs: int = 1
    env_spacing: float = 2.0
    entities: dict[str, EntityCfg] = field(default_factory=dict)
    # we leave sensors and terrains to be added later


@dataclass
class MotrixEntityData:
    root_link_pose_w: torch.Tensor
    root_link_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_acc: torch.Tensor
    joint_pos_target: torch.Tensor
    joint_vel_target: torch.Tensor
    
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


class MotrixScene:
    def __init__(self, cfg: MotrixSceneCfg):
        self.cfg = cfg
        self._entities: dict[str, MotrixEntity] = {}
        msd_scene = mx.msd.from_str(plane_xml)
        for name, cfg in self.cfg.entities.items():
            ent = MotrixEntity(cfg)
            prefix = name + "_"
            ent._prefix = prefix
            msd_scene.attach(
                ent.msd,
                other_translation=ent.cfg.init_state.pos,
                other_rotation=wxyz2xyzw(torch.tensor(ent.cfg.init_state.rot)).tolist(),
                other_prefix=prefix,
            )
            for geom in ("FL", "RL", "FR", "RR"):
                sensor = mx.msd.ContactSensor()
                sensor.name = prefix + geom + "contact"
                sensor.match_ = mx.msd.ContactMatch.geom_pair(
                    f"{name}_{geom}_collision",
                    "floor",
                )
                ent.msd.sensors.contact.append(sensor)
            self._entities[name] = ent
        self.msd_model: mx.SceneModel = msd_scene.build()
        self.msd_data = mx.SceneData(self.msd_model, batch=(self.cfg.num_envs,))
        self.msd_model.step(self.msd_data)

        self._env_origins = compute_env_origins_grid(
            self.cfg.num_envs,
            self.cfg.env_spacing,
        )
        for name, ent in self._entities.items():
            ent.initialize(self.msd_model, self.msd_data)
    
    def update(self, dt: float):
        for ent in self._entities.values():
            ent.update(dt)
    
    @property
    def entities(self) -> dict[str, MotrixEntity]:
        return self._entities
    
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
        return self._entities

    @property
    def sensors(self) -> dict:
        return {}

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
                scene.msd_model,
                batch=scene.num_envs,
                render_offset=scene.render_offsets,
            )

    def get_physics_dt(self) -> float:
        return float(self.scene.msd_model.options.timestep)

    def has_gui(self) -> bool:
        return self._render_app is not None

    def step(self, render: bool = False) -> None:
        self.scene.msd_model.step(self.scene.msd_data)

    def render(self) -> None:
        if self._render_app is not None:
            self._render_app.sync(self.scene.msd_data)

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
        self.mjcf_path = cfg.motrix_mjcf_path_fn(cfg)
        self.msd = mx.msd.from_file(self.mjcf_path)
        self._joint_names: list[str] = []
        self._body_names: list[str] = []
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
            return value.detach().to(device=_DEVICE, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=_DEVICE)

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32, copy=False)    

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
        lin_vel = self._base_link.get_linear_velocity(self._mx_data)
        ang_vel = self._base_link.get_angular_velocity(self._mx_data)
        return torch.cat(
            [self._as_tensor(lin_vel), self._as_tensor(ang_vel)],
            dim=-1,
        )

    def initialize(
        self,
        mx_model: mx.SceneModel,
        mx_data: mx.SceneData,
    ):
        self._mx_model = mx_model
        self._mx_data = mx_data
        body_name = f"{self._prefix}base_link"
        self._body = mx_model.get_body(body_name)
        if self._body is None:
            raise ValueError(f"Body {body_name} not found in model")
        self._base_link = self._body.base_link
        self._floatingbase = self._body.floatingbase
        if self._floatingbase is None:
            raise ValueError(f"Body {body_name} has no floating base")

        # TODO: find better ways to get joint and body names
        self._joint_names = [
            actuator.name.replace(self._prefix, "")
            for actuator in self._body.actuators
        ]
        self._body_names = [
            name.replace(self._prefix, "")
            for name in self._mx_model.link_names if name.startswith(self._prefix)
        ]

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
        self._data = MotrixEntityData(
            root_link_pose_w=pose,
            root_link_vel_w=root_link_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_acc=torch.zeros_like(joint_vel),
            joint_pos_target=default_joint_pos.clone(),
            joint_vel_target=default_joint_vel.clone(),
            default_root_state=root_state,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
        )
    
    def update(self, dt: float):
        self._data.root_link_pose_w = self._as_tensor(self._body.get_pose(self._mx_data))
        self._data.root_link_vel_w = self._read_root_link_vel_w()

        joint_pos = self._as_tensor(self._body.get_joint_dof_pos(self._mx_data))
        joint_vel = self._as_tensor(self._body.get_joint_dof_vel(self._mx_data))
        joint_acc = (joint_vel - self._data.joint_vel) / dt

        self._data.joint_pos = joint_pos
        self._data.joint_vel = joint_vel
        self._data.joint_acc = joint_acc

    @property
    def data(self) -> MotrixEntityData:
        return self._data

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names) # copy the list to avoid modifying the original

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names) # copy the list to avoid modifying the original

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
        self._sync_kinematics()

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
        self._sync_kinematics()

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


class MotrixContactSensor:
    ...



def wxyz2xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz[..., [1, 2, 3, 0]]


def xyzw2wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[..., [3, 0, 1, 2]]
