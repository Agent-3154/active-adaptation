from __future__ import annotations

import torch
import mujoco
import mujoco_warp as mjwarp
from dataclasses import dataclass, replace
from typing import Sequence

from mjlab.entity import Entity
from mjlab.sim import WarpBridge
from mjlab.sensor import Sensor, SensorCfg, ContactData
from mjlab.sensor.contact_sensor import _AirTimeState
from isaaclab.utils.string import resolve_matching_names


@dataclass
class CfrcContactSensorCfg(SensorCfg):
    """
    Compared to mjlab's ContactSensor, this sensor directly reads the 
    per-body contact forces from `data.cfrc_ext`.
    """
    track_air_time: bool = True

    def build(self) -> CfrcContactSensor:
        return CfrcContactSensor(self)


class CfrcContactSensor(Sensor[ContactData]):
    def __init__(self, cfg: CfrcContactSensorCfg) -> None:
        self.cfg = cfg
        self._contact_data = ContactData()
        self._air_time_state = None
    
    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        pass

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: WarpBridge[mjwarp.Model],
        data: WarpBridge[mjwarp.Data],
        device: str,
    ) -> None:
        n_envs = data.time.shape[0]
        b_body = mj_model.nbody
        
        # mj_model.sensor_rne_postconstraint = True
        model.struct.sensor_rne_postconstraint = True

        self.body_names = [mj_model.body(i).name for i in range(b_body)]
        
        self._data: mjwarp.Data = data
        self._device = device
        if self.cfg.track_air_time:
            self._air_time_state = _AirTimeState(
            current_air_time=torch.zeros((n_envs, b_body), device=device),
            last_air_time=torch.zeros((n_envs, b_body), device=device),
            current_contact_time=torch.zeros((n_envs, b_body), device=device),
            last_contact_time=torch.zeros((n_envs, b_body), device=device),
            last_time=torch.zeros((n_envs,), device=device),
        )
    
    @property
    def data(self) -> ContactData:
        return replace(self._contact_data)
        
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._air_time_state is None:
            return

        if env_ids is None:
            env_ids = slice(None)

        # Reset air time state for specified envs.
        self._air_time_state.current_air_time[env_ids] = 0.0
        self._air_time_state.last_air_time[env_ids] = 0.0
        self._air_time_state.current_contact_time[env_ids] = 0.0
        self._air_time_state.last_contact_time[env_ids] = 0.0
        if self._data is not None:
            self._air_time_state.last_time[env_ids] = self._data.time[env_ids]
    
    def update(self, dt: float) -> None:
        
        self._contact_data.force = self._data.cfrc_ext[:, :, :3]
        self._contact_data.torque = self._data.cfrc_ext[:, :, 3:]
        self._contact_data.found = self._contact_data.force.norm(dim=-1) > 0.1

        if self._air_time_state is not None:
            self._update_air_time_tracking()
            self._contact_data.current_air_time = self._air_time_state.current_air_time
            self._contact_data.last_air_time = self._air_time_state.last_air_time
            self._contact_data.current_contact_time = self._air_time_state.current_contact_time
            self._contact_data.last_contact_time = self._air_time_state.last_contact_time
    
    def _update_air_time_tracking(self) -> None:
        current_time = self._data.time
        elapsed_time = current_time - self._air_time_state.last_time
        elapsed_time = elapsed_time.unsqueeze(-1)

        is_contact = self._contact_data.found
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

        state.last_time[:] = current_time


    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, self.body_names, preserve_order)
    
    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_in_contact = self.data.current_contact_time > 0.0
        less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

