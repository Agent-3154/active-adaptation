"""Shared helpers for Warp / simple-raycaster scene sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class WarpSensorSpec:
    """Deferred scene sensor returned by AA sensor factories.

    Native Isaac / mjlab sensors return backend ``*SensorCfg`` objects. Warp
    sensors return a spec that is instantiated after the scene adapter exists.
    """

    name: str
    factory: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def instantiate(self, env: Any) -> Any:
        return self.factory(env=env, name=self.name, **self.kwargs)


def is_warp_sensor_spec(obj: Any) -> bool:
    return isinstance(obj, WarpSensorSpec)


def install_warp_sensors(
    scene: Any,
    specs: Mapping[str, WarpSensorSpec],
    *,
    env: Any,
) -> None:
    """Build warp sensors and attach them to the scene adapter."""
    if not specs:
        return
    sensors = {name: spec.instantiate(env) for name, spec in specs.items()}
    for sensor in sensors.values():
        sensor.initialize()
    scene._warp_sensors = sensors


def update_warp_sensors(scene: Any) -> None:
    """Refresh all warp scene sensors (called once per control step)."""
    for sensor in getattr(scene, "_warp_sensors", {}).values():
        sensor.update()


def merge_sensors(native: dict, warp: dict | None) -> dict:
    if not warp:
        return native
    return {**native, **warp}
