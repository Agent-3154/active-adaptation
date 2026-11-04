"""Expose Warp's separate convex-solver capacity on mjlab 1.6.

Broadphase candidates and actual EPA solves need not have equal capacities.
Keep mjlab's initialization order; only the put_data allocation gains nccdmax.
Neither collision filtering nor the contact solver is changed.
"""
import mujoco_warp as mjwarp
import warp as wp
from mjlab.sim import Simulation
from mjlab.sim.sim_data import TorchArray, WarpBridge
from mjlab.utils.nan_guard import NanGuard


class CapacitySimulation(Simulation):
    def __init__(self, *, nccdmax=None, **kwargs):
        if nccdmax is not None and nccdmax <= 0:
            raise ValueError("nccdmax must be positive")
        self._aa_nccdmax = nccdmax
        super().__init__(**kwargs)

    def _finish_init(self):
        if self._aa_nccdmax is None:
            return super()._finish_init()
        self._wp_data = mjwarp.put_data(
            self._mj_model, self._mj_data, nworld=self.num_envs,
            nconmax=self.cfg.nconmax, njmax=self.cfg.njmax,
            nccdmax=self._aa_nccdmax,
        )
        self._reset_mask_wp = wp.zeros(self.num_envs, dtype=bool)
        self._reset_mask = TorchArray(self._reset_mask_wp)
        self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
        self._data_bridge = WarpBridge(self._wp_data)
        self._sensor_context = None
        self.use_cuda_graph = self._should_use_cuda_graph()
        self.create_graph()
        self.nan_guard = NanGuard(self.cfg.nan_guard, self.num_envs, self._mj_model)
