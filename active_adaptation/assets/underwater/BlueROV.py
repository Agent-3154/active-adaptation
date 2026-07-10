from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from active_adaptation.envs.robots.underwater import HydrodynamicsCfg, UnderwaterRobot
from active_adaptation.registry import Registry
from active_adaptation import ROBOT_MODEL_DIR

registry = Registry.instance()

USD_PATH = ROBOT_MODEL_DIR / "underwater" / "BlueROV.usd"

# --- hydrodynamics (from BlueROV.yaml) ---
DRAG_COEF = 0.3
VOLUME = 0.0113459
COBM = 0.01
ADDED_MASS = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
LINEAR_DAMPING = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
QUADRATIC_DAMPING = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

# --- rotor / T200 thruster parameters (from BlueROV.yaml) ---
NUM_ROTORS = 6
ROTOR_DIRECTIONS = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
ROTOR_TIME_CONSTANTS = {f"rotor_{i}": 0.01 for i in range(NUM_ROTORS)}
ROTOR_FORCE_CONSTANTS = {f"rotor_{i}": 4.4e-07 for i in range(NUM_ROTORS)}
ROTOR_MAX_ROTATION_VEL_RPM = (3900.0,) * 6
ROTOR_MOMENT_CONSTANTS = (1.3677728816219314e-09,) * 6
ROTOR_MAX_ROTATION_VEL_RAD_S = tuple(
    rpm * 2.0 * math.pi / 60.0 for rpm in ROTOR_MAX_ROTATION_VEL_RPM
)

INIT_POS = (0.0, 0.0, 2.0)

JOINT_NAMES_SIMULATION = [f"rotor_{i}" for i in range(NUM_ROTORS)]
BODY_NAMES_SIMULATION = ["base_link", *[f"rotor_{i}" for i in range(NUM_ROTORS)]]



def make_isaaclab_cfg(self_collisions: bool = False):
    from active_adaptation.assets.asset_cfg import (
        AssetSpec,
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )

    asset_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Thruster dynamics are applied as body wrenches; joints are free-spinning placeholders.
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["rotor_.*"],
                effort_limit_sim=0.0,
                velocity_limit_sim=max(ROTOR_MAX_ROTATION_VEL_RAD_S),
                stiffness=0.0,
                damping=0.0,
            ),
        },
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    return AssetSpec(
        config=asset_cfg,
        sensors={},
        # Wrapper is created as an instance and initialized by backend env.
        wrapper=UnderwaterRobot(
            cfg=HydrodynamicsCfg(
                volume=VOLUME,
                coBM=COBM,
                added_mass=ADDED_MASS,
                linear_damping=LINEAR_DAMPING,
                quadratic_damping=QUADRATIC_DAMPING,
            ),
            rotor_time_constants=ROTOR_TIME_CONSTANTS,
            rotor_force_constants=ROTOR_FORCE_CONSTANTS,
        ),
    )


def make_mjlab_cfg(motrix: bool = False):
    raise NotImplementedError("MJLab backend is not supported for BlueROV")


def make_cfg(backend: Literal["isaaclab", "mjlab", "motrix"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    elif backend == "mjlab":
        return make_mjlab_cfg(motrix=False)
    elif backend == "motrix":
        return make_mjlab_cfg(motrix=True)
    else:
        raise ValueError(f"Invalid backend: {backend}")


registry.register("asset", "bluerov", make_cfg)
