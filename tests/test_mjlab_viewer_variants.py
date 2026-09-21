from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from active_adaptation.envs.backends.mjlab.adapter import MjlabSimAdapter
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


def test_offscreen_renderer_receives_per_world_model_fields() -> None:
    sim = SimpleNamespace(
        mj_model=object(),
        model=object(),
        expanded_fields={"geom_dataid", "body_mass"},
    )
    cfg = object()
    scene = object()

    with patch(
        "mjlab.viewer.offscreen_renderer.OffscreenRenderer"
    ) as renderer_cls:
        renderer = MjlabSimAdapter(sim, viewer_cfg=cfg, scene=scene)._get_offscreen_renderer()

    renderer_cls.assert_called_once_with(
        model=sim.mj_model,
        cfg=cfg,
        scene=scene,
        sim_model=sim.model,
        expanded_fields=sim.expanded_fields,
    )
    renderer.initialize.assert_called_once_with()
