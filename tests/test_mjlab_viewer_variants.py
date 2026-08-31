from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer


def test_viewer_receives_per_world_model_fields() -> None:
    env = SimpleNamespace(num_envs=16)
    sim = SimpleNamespace(
        mj_model=object(),
        model=object(),
        expanded_fields={"geom_dataid", "body_mass"},
    )
    server = MagicMock()
    scene = MagicMock()

    with (
        patch(
            "active_adaptation.envs.backends.mjlab.viewer.viser.ViserServer",
            return_value=server,
        ),
        patch(
            "active_adaptation.envs.backends.mjlab.viewer.ViserMujocoScene",
            return_value=scene,
        ) as scene_cls,
    ):
        MjLabViewer(env, sim).setup()

    scene_cls.assert_called_once_with(
        server,
        sim.mj_model,
        16,
        sim_model=sim.model,
        expanded_fields=sim.expanded_fields,
    )
