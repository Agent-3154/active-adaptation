from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry
from typing import Literal

registry = Registry.instance()

# currently, only isaac backend is supported for dummy objects

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


def make_dummy_stand(backend: Literal["isaaclab", "mjlab"]):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_stand")


def make_dummy_basket(backend: Literal["isaaclab", "mjlab"]):
    if backend != "isaaclab":
        raise NotImplementedError
    return _make_rigid("dummy_basket")


registry.register("asset", "dummy_stand", make_dummy_stand)
registry.register("asset", "dummy_basket", make_dummy_basket)
