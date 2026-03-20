import torch
import viser
import numpy as np
from mjlab.sim import Simulation
from mjlab.viewer.viser import ViserMujocoScene

from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.utils.profiling import ScopedTimer


class MjLabViewer:
    """
    Different from `mjlab.viewer.viser_play.ViserPlayViewer`, this
    viewer is not responsible for stepping the environment and is updated
    synchronously from the environment step loop.
    """

    def __init__(self, env: _EnvBase):
        self.env = env
        self.sim: Simulation = env.sim

        self._server = viser.ViserServer(label="mjlab")
        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return

        self._scene = ViserMujocoScene.create(
            server=self._server,
            mj_model=self.sim.mj_model,
            num_envs=self.env.num_envs,
        )
        self._scene.camera_tracking_enabled = False
        self._scene.debug_visualization_enabled = True
        self._scene.show_all_envs = True
        self._scene.env_idx = 0

        tabs = self._server.gui.add_tab_group()
        self._scene.create_groups_gui(tabs)
        self._scene.create_visualization_gui()
        self._is_setup = True

    @property
    def scene(self) -> ViserMujocoScene | None:
        return getattr(self, "_scene", None)

    def add_batched_axes(self, name: str):
        axes_handle = self._server.scene.add_batched_axes(
            name=name,
            batched_wxyzs=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 4
            ),
            batched_positions=torch.tensor([[0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 3
            ),
            batched_scales=torch.tensor([[1.0, 1.0, 1.0]]).expand(
                self.env.num_envs, 3
            ),
        )
        return axes_handle

    def add_line_segments(
        self, name: str, colors: tuple[float, float, float] | torch.Tensor
    ):
        lines_handle = self._server.scene.add_line_segments(
            name=name,
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 2, 3
            ),
            colors=colors,
        )
        return lines_handle

    def clear(self):
        if self._scene is None:
            return
        self._scene.clear()

    def _update_selected_env(self):
        scene = self._scene
        if scene is None:
            raise RuntimeError("MjLab viewer is not set up.")

        env_idx = int(scene.env_idx)
        body_xpos = self.sim.data.xpos[env_idx : env_idx + 1].cpu().numpy()
        body_xmat = self.sim.data.xmat[env_idx : env_idx + 1].cpu().numpy()
        if scene.mj_model.nmocap > 0:
            mocap_pos = self.sim.data.mocap_pos[env_idx : env_idx + 1].cpu().numpy()
            mocap_quat = self.sim.data.mocap_quat[env_idx : env_idx + 1].cpu().numpy()
        else:
            mocap_pos = np.zeros((1, 0, 3))
            mocap_quat = np.zeros((1, 0, 4))

        scene_offset = np.zeros(3)
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            tracked_pos = body_xpos[0, scene._tracked_body_id, :].copy()
            scene_offset = -tracked_pos

        contacts = None
        if scene.show_contact_points or scene.show_contact_forces:
            scene.mj_data.qpos[:] = self.sim.data.qpos[env_idx].cpu().numpy()
            scene.mj_data.qvel[:] = self.sim.data.qvel[env_idx].cpu().numpy()
            if scene.mj_model.nmocap > 0:
                scene.mj_data.mocap_pos[:] = mocap_pos[0]
                scene.mj_data.mocap_quat[:] = mocap_quat[0]
            import mujoco

            mujoco.mj_forward(scene.mj_model, scene.mj_data)
            contacts = scene._extract_contacts_from_mjdata(scene.mj_data)

        scene._update_visualization(
            body_xpos,
            body_xmat,
            mocap_pos,
            mocap_quat,
            0,
            scene_offset,
            contacts,
        )
        scene._sync_debug_visualizations(scene_offset)

    def update(self):
        if self._scene is None:
            raise RuntimeError("MjLab viewer is not set up.")
        if self._scene.show_only_selected and self.env.num_envs > 1:
            with ScopedTimer("viewer.update.selected_fast_path", sync=False):
                self._update_selected_env()
        else:
            with ScopedTimer("viewer.update.scene_update", sync=False):
                with self._server.atomic():
                    self._scene.update(self.sim.data)
                    self._server.flush()

    def close(self):
        self._server.stop()
