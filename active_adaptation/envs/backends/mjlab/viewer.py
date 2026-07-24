import torch
import viser
import numpy as np
from mjlab.sim import Simulation
from mjlab.viewer.viser import ViserMujocoScene

from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.utils.profiling import ScopedTimer


def _rgba_to_rgb255(color: tuple[float, ...] | list[float]) -> tuple[int, int, int]:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if max(r, g, b) <= 1.0:
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


class MjLabViewer:
    """
    Different from `mjlab.viewer.viser.viewer.ViserPlayViewer`, this
    viewer is not responsible for stepping the environment and is updated
    synchronously from the environment step loop.
    """

    def __init__(self, env: _EnvBase, sim: Simulation):
        self.env = env
        self.sim = sim

        self._server = viser.ViserServer(label="mjlab")
        self._is_setup = False

        self._cameras: dict[str, viser.CameraFrustumHandle] = {}
        self._line_handle = None
        self._point_handle = None
        self._debug_line_pts: list[np.ndarray] = []
        self._debug_line_cols: list[np.ndarray] = []
        self._debug_point_pts: list[np.ndarray] = []
        self._debug_point_cols: list[np.ndarray] = []
        self._debug_point_size: float = 0.02

    def setup(self):
        if self._is_setup:
            return

        self._scene = ViserMujocoScene(
            self._server,
            self.sim.mj_model,
            self.env.num_envs,
        )
        self._scene.debug_visualization_enabled = True
        self._scene.camera_tracking_enabled = False
        self._scene.show_all_envs = True
        self._scene.env_idx = 0

        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Scene", icon=viser.Icon.SETTINGS):
            self._scene.create_scene_gui()
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            self._scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            self._scene.create_groups_gui()
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

    # ------------------------------------------------------------------
    # MDP debug primitives (vectors / points), synced in update()
    # ------------------------------------------------------------------

    def clear_debug(self) -> None:
        self._debug_line_pts.clear()
        self._debug_line_cols.clear()
        self._debug_point_pts.clear()
        self._debug_point_cols.clear()

    def vector(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        v_np = v.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape != v_np.shape:
            raise ValueError(f"x and v must match, got {x_np.shape} and {v_np.shape}")
        seg = np.stack([x_np, x_np + v_np], axis=1)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def point(
        self,
        x: torch.Tensor,
        color: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0),
        size: float = 10.0,
    ) -> None:
        pts = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (pts.shape[0], 3)).copy()
        self._debug_point_pts.append(pts)
        self._debug_point_cols.append(cols)
        self._debug_point_size = max(float(size) * 0.002, 0.005)

    def plot(
        self,
        x: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape[0] < 2:
            return
        seg = np.stack([x_np[:-1], x_np[1:]], axis=1)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def _sync_debug_geometry(self) -> None:
        if self._debug_line_pts:
            points = np.concatenate(self._debug_line_pts, axis=0)
            colors = np.concatenate(self._debug_line_cols, axis=0)
        else:
            points = np.zeros((0, 2, 3), dtype=np.float32)
            colors = np.zeros((0, 2, 3), dtype=np.uint8)

        if self._line_handle is None:
            init_pts = points if points.shape[0] > 0 else np.zeros((1, 2, 3), dtype=np.float32)
            init_cols = colors if colors.shape[0] > 0 else np.zeros((1, 2, 3), dtype=np.uint8)
            self._line_handle = self._server.scene.add_line_segments(
                "/debug/mdp_lines",
                init_pts,
                init_cols,
                line_width=2.0,
                visible=points.shape[0] > 0,
            )
        elif points.shape[0] == 0:
            self._line_handle.visible = False
        else:
            self._line_handle.points = points
            self._line_handle.colors = colors
            self._line_handle.visible = True

        if self._debug_point_pts:
            pts = np.concatenate(self._debug_point_pts, axis=0)
            cols = np.concatenate(self._debug_point_cols, axis=0)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
            cols = np.zeros((0, 3), dtype=np.uint8)

        if self._point_handle is None:
            init_pts = pts if pts.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)
            init_cols = cols if cols.shape[0] > 0 else np.zeros((1, 3), dtype=np.uint8)
            self._point_handle = self._server.scene.add_point_cloud(
                "/debug/mdp_points",
                init_pts,
                init_cols,
                point_size=self._debug_point_size,
                visible=pts.shape[0] > 0,
            )
        elif pts.shape[0] == 0:
            self._point_handle.visible = False
        else:
            self._point_handle.points = pts
            self._point_handle.colors = cols
            self._point_handle.point_size = self._debug_point_size
            self._point_handle.visible = True

    # ------------------------------------------------------------------
    # Camera frustums
    # ------------------------------------------------------------------

    def register_camera(
        self,
        name: str,
        *,
        fov_y: float,
        aspect: float,
        scale: float = 0.15,
    ):
        """Create a Viser camera frustum (OpenCV +Z forward)."""
        if name in self._cameras:
            return self._cameras[name]
        handle = self._server.scene.add_camera_frustum(
            f"/cameras/{name}",
            fov=float(fov_y),
            aspect=float(aspect),
            scale=float(scale),
            color=(200, 200, 200),
            format="jpeg",
        )
        self._cameras[name] = handle
        return handle

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
        self._sync_debug_geometry()
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
