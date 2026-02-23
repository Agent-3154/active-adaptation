import viser
import time
import threading
import torch

from mjlab.sim import Simulation
from mjlab.viewer.viser import ViserMujocoScene
from mjlab.viewer.base import Timer
from active_adaptation.envs.env_base import _EnvBase


class MjLabViewer:
    """
    Different from `mjlab.viewer.viser_play.ViserPlayViewer`, this
    viewer is not responsible for stepping the environment.
    """
    def __init__(self, env: _EnvBase):
        self.env = env
        self.sim: Simulation = env.sim
        self.frame_rate = 30.0
        self.frame_time = 1.0 / self.frame_rate

        self._server = viser.ViserServer(label="mjlab")
        self._timer = Timer()
        self._render_timer = Timer()
        self._time_until_next_frame = 0.0
        self._thread = None

    def setup(self):
        self._scene = ViserMujocoScene.create(
            server=self._server,
            mj_model=self.sim.mj_model,
            num_envs=self.env.num_envs,
        )
        self._scene.env_idx = 0

        tabs = self._server.gui.add_tab_group()
        self._scene.create_groups_gui(tabs)
        self._scene.create_visualization_gui()
    
    def add_batched_axes(self, name: str):
        axes_handle = self._server.scene.add_batched_axes(
            name=name,
            batched_wxyzs=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(self.env.num_envs, 4),
            batched_positions=torch.tensor([[0.0, 0.0, 0.0]]).expand(self.env.num_envs, 3),
            batched_scales=torch.tensor([[1.0, 1.0, 1.0]]).expand(self.env.num_envs, 3),
        )
        return axes_handle
    
    def add_line_segments(self, name: str, colors: tuple[float, float, float] | torch.Tensor):
        lines_handle = self._server.scene.add_line_segments(
            name=name,
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).expand(self.env.num_envs, 2, 3),
            colors=colors,
        )
        return lines_handle
    
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()
    
    def tick(self) -> bool:
        elapsed_time = self._timer.tick()
        self._time_until_next_frame -= elapsed_time

        if self._time_until_next_frame > 0:
            return False
        
        self._time_until_next_frame += self.frame_time
        if self._time_until_next_frame < -self.frame_time:
            self._time_until_next_frame = 0.0
        
        with self._render_timer.measure_time():
            with self._server.atomic():
                self._scene.update(self.sim.data)
                self._server.flush()
        
        return True
    
    def close(self):
        self._server.stop()

    def _run_loop(self):
        self.setup()
        try:
            while self.is_running():
                if not self.tick():
                    time.sleep(0.001)
        finally:
            self.close()

    def run_async(self):
        """Start the viewer in a background thread. Returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Viewer is already running")
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the viewer loop."""
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
