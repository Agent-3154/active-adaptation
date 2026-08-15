from __future__ import annotations

from typing import Sequence

from active_adaptation.registry import Registry

registry = Registry.instance()

# Isaac scene key ``robot`` is spawned at ``{ENV_REGEX_NS}/Robot``; objects use the YAML key.
_ISAAC_ENTITY_PRIM = {"robot": "Robot"}

# TerrainImporter (prim_path="/World/ground") always names the child ``terrain``.
# Contact filters must hit the collision geom, not that Xform. From Isaac Lab
# ``test_contact_sensor``: plane → GroundPlane/CollisionPlane, generator → mesh.
_ISAAC_TERRAIN_FILTER_PRIMS = (
    "/World/ground/terrain/GroundPlane/CollisionPlane",
    "/World/ground/terrain/mesh",
)


def _isaac_prim(entity: str) -> str:
    return _ISAAC_ENTITY_PRIM.get(entity, entity)


def _isaac_prim_path(entity: str, pattern: str | None) -> str:
    """Isaac contact ``prim_path``.

    Articulations (``robot``) put ContactReportAPI on *child* links, so the
    default is ``{ENV}/Robot/.*``. A RigidObject is usually the rigid body
    itself (``{ENV}/object``); ``{ENV}/object/.*`` only sees visuals/collisions
    and raises "could not find any bodies with contact reporter API".
    """
    root = f"{{ENV_REGEX_NS}}/{_isaac_prim(entity)}"
    if not pattern:
        return root
    return f"{root}/{pattern}"


def _resolve_body_pattern(entity: str, pattern: str | None) -> tuple[str | None, str]:
    """Return ``(isaac_pattern, mjlab_pattern)``.

    ``pattern is None``: robot → all child links; other entities → entity root
    (Isaac) / all bodies (mjlab).
    """
    if pattern is None:
        return (".*" if entity == "robot" else None), ".*"
    return (pattern or None), (pattern or ".*")


def contact_sensor(
    backend: str,
    name: str,
    entity: str = "robot",
    pattern: str | None = None,
    secondary_entity: str | None = None,
    secondary_pattern: str | None = None,
    track_air_time: bool = True,
    history_length: int = 3,
    fields: Sequence[str] = ("found", "force"),
    reduce: str = "netforce",
):
    """Build a backend contact-sensor cfg stored as ``scene.sensors[name]``.

    Shared kwargs (task YAML under ``sensors.<name>``)::

        sensors:
          contact_forces:
            _target_: contact_sensor
            entity: robot          # scene entity key
            pattern: ".*"          # bodies; omit for a rigid object (entity root)
            secondary_entity: terrain   # optional contact filter
            track_air_time: true
            history_length: 3

    ``entity`` / ``secondary_entity`` are scene keys (``robot``, object YAML keys).
    Isaac maps ``robot`` → prim ``Robot``; other keys are used as prim names.
    Default ``pattern`` is ``.*`` (child links) for ``robot`` and the entity root
    for rigid objects — Isaac puts ContactReportAPI on the object prim itself.

    ``secondary_entity: terrain`` filters against the ground:

    * mjlab: literal body ``terrain`` (``ContactMatch.entity=None``).
    * Isaac: ``/World/ground/terrain/GroundPlane/CollisionPlane`` (plane) and
      ``/World/ground/terrain/mesh`` (generator). Override with ``secondary_pattern``
      if you need a different prim. Filtered forces are ``force_matrix_w``, not
      ``net_forces_w``.
    """
    isaac_pattern, mjlab_pattern = _resolve_body_pattern(entity, pattern)
    if backend in ("isaac", "isaaclab"):
        from isaaclab.sensors import ContactSensorCfg

        filter_prim_paths_expr: list[str] = []
        if secondary_entity == "terrain":
            if secondary_pattern is not None:
                filter_prim_paths_expr = [secondary_pattern]
            else:
                filter_prim_paths_expr = list(_ISAAC_TERRAIN_FILTER_PRIMS)
        elif secondary_entity is not None:
            filter_prim_paths_expr = [
                _isaac_prim_path(secondary_entity, secondary_pattern or ".*")
            ]
        return ContactSensorCfg(
            prim_path=_isaac_prim_path(entity, isaac_pattern),
            track_air_time=track_air_time,
            history_length=history_length,
            filter_prim_paths_expr=filter_prim_paths_expr,
        )

    if backend == "mjlab":
        from mjlab.sensor import ContactMatch, ContactSensorCfg

        if secondary_entity is None and secondary_pattern is None:
            secondary = None
        elif secondary_entity in (None, "terrain"):
            secondary = ContactMatch(
                mode="body",
                pattern=secondary_pattern or "terrain",
                entity=None,
            )
        else:
            secondary = ContactMatch(
                mode="body",
                pattern=secondary_pattern or ".*",
                entity=secondary_entity,
            )
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="body", pattern=mjlab_pattern, entity=entity),
            secondary=secondary,
            fields=tuple(fields),
            reduce=reduce,
            num_slots=1,
            track_air_time=track_air_time,
            history_length=history_length,
        )

    raise ValueError(f"Unknown backend: {backend}")


registry.register("sensor", "contact_sensor", contact_sensor)
