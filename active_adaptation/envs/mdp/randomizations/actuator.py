from __future__ import annotations

import torch
from typing import Optional
from typing_extensions import override

import active_adaptation
from active_adaptation.envs.mdp.randomizations.base import Randomization
from active_adaptation.envs.mdp.randomizations.common import NestedRangeType

if active_adaptation.get_backend() == "isaac":
    from isaaclab.utils.string import resolve_matching_names_values
if active_adaptation.get_backend() == "mjlab":
    from mjlab.utils.lab_api.string import resolve_matching_names_values
    from mjlab.sim import Simulation
    from mjlab.actuator import BuiltinPositionActuator


class actuator_pd_gains(Randomization):
    """Randomize PD stiffness and damping gains.

    For the **mjlab** backend, only ``BuiltinPositionActuator`` is supported: we
    write ``actuator_gainprm`` / ``actuator_biasprm`` in the same layout as
    mjlab's ``pd_gains`` ``scale`` path. Other actuator types (e.g.
    ``IdealPdActuator`` or non-position XML actuators) are not handled.

    For **isaac**, gains are applied via the articulation joint stiffness and
    damping write APIs (not MuJoCo actuators).
    """

    supported_backends = ("isaac", "mjlab")
    
    mj_fields = (
        "actuator_gainprm",
        "actuator_biasprm",
    )

    def __init__(
        self,
        env,
        stiffness_range: Optional[NestedRangeType] = None,
        damping_range: Optional[NestedRangeType] = None,
    ):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.stiffness_range = dict(stiffness_range) if stiffness_range is not None else None
        self.damping_range = dict(damping_range) if damping_range is not None else None

        if self.env.backend == "mjlab":
            self._init_mjlab()
        elif self.env.backend == "isaac":
            self._init_isaac()

    def _init_isaac(self):
        if self.stiffness_range is not None:
            ids, _, value = resolve_matching_names_values(
                self.stiffness_range, self.asset.joint_names
            )
            self.stiffness_id = torch.tensor(ids, device=self.device)
            self.stiffness_default = self.asset.data.joint_stiffness[0, self.stiffness_id]
            low, high = torch.tensor(value, device=self.device).unbind(1)
            self.stiffness_low = low * self.stiffness_default
            self.stiffness_scale = (high - low) * self.stiffness_default

        if self.damping_range is not None:
            ids, _, value = resolve_matching_names_values(
                self.damping_range, self.asset.joint_names
            )
            self.damping_id = torch.tensor(ids, device=self.device)
            self.damping_default = self.asset.data.joint_damping[0, self.damping_id]
            low, high = torch.tensor(value, device=self.device).unbind(1)
            self.damping_low = low * self.damping_default
            self.damping_scale = (high - low) * self.damping_default

    def _init_mjlab(self):
        sim: Simulation = self.env.sim
        self.model = sim.model

        for actuator in self.asset.actuators:
            if not isinstance(actuator, BuiltinPositionActuator):
                raise ValueError(f"Actuator {actuator} is not a BuiltinPositionActuator")

        if self.stiffness_range is not None:
            kp_ids, _, kp_ranges = resolve_matching_names_values(
                self.stiffness_range, self.asset.actuator_names
            )
            assert len(kp_ids) > 0, f"No actuator IDs found for stiffness range {self.stiffness_range!r}"
            self.kp_ctrl_ids = self.asset.indexing.ctrl_ids[kp_ids]
            default_gainprm = sim.get_default_field("actuator_gainprm")
            default_biasprm = sim.get_default_field("actuator_biasprm")
            self.kp_gain_def = default_gainprm[self.kp_ctrl_ids, 0]
            self.kp_bias_def = default_biasprm[self.kp_ctrl_ids, 1]
            kp_low, kp_high = torch.tensor(kp_ranges, device=self.device).unbind(1)
            self.kp_low = kp_low
            self.kp_high = kp_high
        else:
            self.kp_ctrl_ids = None

        if self.damping_range is not None:
            kd_ids, _, kd_ranges = resolve_matching_names_values(
                self.damping_range, self.asset.actuator_names
            )
            assert len(kd_ids) > 0, f"No actuator IDs found for damping range {self.damping_range!r}"
            self.kd_ctrl_ids = self.asset.indexing.ctrl_ids[kd_ids]
            default_biasprm = sim.get_default_field("actuator_biasprm")
            self.kd_bias_def = default_biasprm[self.kd_ctrl_ids, 2]
            kd_low, kd_high = torch.tensor(kd_ranges, device=self.device).unbind(1)
            self.kd_low = kd_low
            self.kd_high = kd_high
        else:
            self.kd_ctrl_ids = None

    @override
    def reset(self, env_ids):
        if self.env.backend == "mjlab":
            if self.kp_ctrl_ids is not None:
                rand = torch.rand(len(env_ids), len(self.kp_ctrl_ids), device=self.device)
                kp_samples = rand * (self.kp_high - self.kp_low) + self.kp_low
                kp_gain = self.kp_gain_def.unsqueeze(0) * kp_samples
                kp_bias = self.kp_bias_def.unsqueeze(0) * kp_samples
                self.model.actuator_gainprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 0] = kp_gain
                self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 1] = kp_bias
            if self.kd_ctrl_ids is not None:
                rand = torch.rand(len(env_ids), len(self.kd_ctrl_ids), device=self.device)
                kd_samples = rand * (self.kd_high - self.kd_low) + self.kd_low
                kd_bias = self.kd_bias_def.unsqueeze(0) * kd_samples
                self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kd_ctrl_ids, 2] = kd_bias
        elif self.env.backend == "isaac":
            if self.stiffness_range is not None:
                rand = torch.rand(len(env_ids), len(self.stiffness_id), device=self.device)
                stiffness = rand * self.stiffness_scale + self.stiffness_low
                self.asset.write_joint_stiffness_to_sim(stiffness, self.stiffness_id, env_ids)

            if self.damping_range is not None:
                rand = torch.rand(len(env_ids), len(self.damping_id), device=self.device)
                damping = rand * self.damping_scale + self.damping_low
                self.asset.write_joint_damping_to_sim(damping, self.damping_id, env_ids)

