from active_adaptation.envs import mdp
from active_adaptation.envs.env_base import _EnvBase


class _Component(mdp.MDPComponent):
    def startup(self):
        pass

    def reset(self, env_ids, tensordict):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass


class _Harness:
    _add_mdp_component = _EnvBase._add_mdp_component

    def __init__(self):
        self._scene_components = []
        self._callback_component_ids = set()
        self._startup_callbacks = [lambda: None]
        self._reset_callbacks = []
        self._pre_step_callbacks = []
        self._post_step_callbacks = []
        self._update_callbacks = []
        self._debug_draw_callbacks = []


def test_mdp_callbacks_register_once_without_clearing_setup_callbacks():
    harness = _Harness()
    component = _Component()

    harness._add_mdp_component(component)
    harness._add_mdp_component(component)

    assert len(harness._scene_components) == 1
    assert len(harness._startup_callbacks) == 2
    assert len(harness._reset_callbacks) == 1
    assert len(harness._update_callbacks) == 1
    assert len(harness._debug_draw_callbacks) == 1
