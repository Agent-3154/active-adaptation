"""Actuator-specific randomization compatibility entry points."""

from __future__ import annotations

from typing import Optional

from active_adaptation.envs.mdp.randomizations.common import (
    NestedRangeType,
    actuator_params,
)


class actuator_pd_gains(actuator_params):
    """Compatibility wrapper for gain-only actuator randomization."""

    def __init__(
        self,
        env,
        stiffness_range: Optional[NestedRangeType] = None,
        damping_range: Optional[NestedRangeType] = None,
    ):
        super().__init__(
            env,
            stiffness_range=stiffness_range,
            damping_range=damping_range,
        )
