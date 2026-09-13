"""Legacy USD-file dummy props (stand, basket) and re-exports.

Procedural furniture lives in ``furnitures``; the grasp board in ``grasp_board``.
Importing this module still registers those assets (via side-effect imports)
plus the legacy Isaac-only USD props below.
"""

from __future__ import annotations

from typing import Literal

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry

# Register procedural assets.
from active_adaptation.assets import furnitures as _furnitures  # noqa: F401
from active_adaptation.assets import grasp_board as _grasp_board  # noqa: F401

# Backward-compatible re-exports.
from active_adaptation.assets.furnitures import (  # noqa: F401
    make_chair,
    make_door,
    make_drawer,
    make_table,
)
from active_adaptation.assets.grasp_board import (  # noqa: F401
    build_grasp_board_spec,
    make_grasp_board,
)

registry = Registry.instance()

Backend = Literal["isaaclab", "mjlab"]


def _make_rigid(name: str):
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    path = ROBOT_MODEL_DIR / "dummy_objects" / f"{name}.usda"

    return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.001,
                angular_damping=0.001,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _make_static_platform(size: tuple[float, float, float] = (0.2, 0.2, 0.2)):
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    return RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.32, 0.32),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, size[2] * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_dummy_stand(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_stand")


def make_dummy_basket(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_basket")


def make_dummy_basket_platform(backend: Backend):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_static_platform()


registry.register("asset", "dummy_stand", make_dummy_stand)
registry.register("asset", "dummy_basket", make_dummy_basket)
registry.register("asset", "dummy_basket_platform", make_dummy_basket_platform)
