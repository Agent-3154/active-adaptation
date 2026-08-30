from __future__ import annotations

from typing import Sequence

from active_adaptation.registry import Registry

registry = Registry.instance()

# Isaac scene key ``robot`` is spawned at ``{ENV_REGEX_NS}/Robot``; objects use the YAML key.
_ISAAC_ENTITY_PRIM = {"robot": "Robot"}

# TerrainImporter (prim_path="/World/ground") always names the child ``terrain``.
# Contact filters must hit the collision geom, not that Xform.
# Default is the *plane* leaf (AA's usual ``terrain: plane``). Listing both plane
# and generator mesh made ``secondary: [terrain, object]`` allocate M=3 and
# PhysX warn that ``.../mesh`` matched 0 entries on plane terrains.
# Override with ``secondary_pattern: /World/ground/terrain/mesh`` for generators.
_ISAAC_TERRAIN_FILTER_PRIM_PLANE = (
    "/World/ground/terrain/GroundPlane/CollisionPlane"
)
_ISAAC_TERRAIN_FILTER_PRIM_MESH = "/World/ground/terrain/mesh"


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


def _as_str_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _normalize_secondaries(
    secondary: str | Sequence[str] | None,
    secondary_pattern: str | Sequence[str] | None,
) -> list[tuple[str, str | None]]:
    """Normalize secondary partners to ``[(entity, pattern), ...]``.

    ``secondary`` is a scene key or list of keys. Patterns align by index when
    ``secondary`` is a list; a single string pattern applies to every partner.
    """
    entities = _as_str_list(secondary)
    if not entities:
        if secondary_pattern is not None:
            raise ValueError("secondary_pattern requires secondary.")
        return []

    patterns = _as_str_list(secondary_pattern)
    if not patterns:
        return [(e, None) for e in entities]
    if len(patterns) == 1:
        return [(e, patterns[0]) for e in entities]
    if len(patterns) != len(entities):
        raise ValueError(
            f"secondary_pattern length ({len(patterns)}) must be 1 or match "
            f"secondary length ({len(entities)})."
        )
    return list(zip(entities, patterns))


def _isaac_filter_paths(
    secondaries: list[tuple[str, str | None]],
) -> list[str]:
    paths: list[str] = []
    for entity, pattern in secondaries:
        if entity == "terrain":
            if pattern is not None:
                paths.append(pattern)
            else:
                paths.append(_ISAAC_TERRAIN_FILTER_PRIM_PLANE)
            continue
        # Filter against another entity: default to all child links for robot,
        # entity root for rigid objects (same rule as primary prim_path).
        isaac_pat, _ = _resolve_body_pattern(entity, pattern)
        paths.append(_isaac_prim_path(entity, isaac_pat))
    return paths


def _isaac_is_multi_body_robot(entity: str, isaac_pattern: str | None) -> bool:
    """True when the Isaac primary would match many ContactReport links."""
    return entity == "robot" and (isaac_pattern is None or isaac_pattern == ".*")


def _isaac_primary_and_filters(
    entity: str,
    isaac_pattern: str | None,
    secondaries: list[tuple[str, str | None]],
) -> tuple[str, list[str]]:
    """Isaac filtered reporting is **one primary body × many filters**.

    YAML may say ``entity: robot`` + ``secondary: object`` (robot↔rigid object).
    With default ``Robot/.*`` that is many primaries × one per-env filter and
    PhysX fails (``expected num_bodies*num_envs, found num_envs``). Auto-invert
    to the supported layout: object as primary, ``Robot/.*`` as filters →
    ``force_matrix_w`` ``(N, 1, B_robot, 3)`` (per-link forces on dim ``M``).

    ``secondary: terrain`` is **not** inverted: terrain is one global collision
    prim and many robot links × one ground filter works (Isaac foot-vs-ground).
    Robot keeps ContactReportAPI; terrain stays a filter only.

    mjlab does not need this flip (many primaries vs one secondary is fine).
    """
    filters = _isaac_filter_paths(secondaries)
    prim = _isaac_prim_path(entity, isaac_pattern)

    if not (
        _isaac_is_multi_body_robot(entity, isaac_pattern)
        and len(secondaries) == 1
    ):
        return prim, filters

    sec_entity, sec_pattern = secondaries[0]
    robot_filter = _isaac_prim_path("robot", ".*")
    if sec_entity == "terrain":
        # Keep robot as primary: terrain is a single global filter prim
        # (/World/ground/.../CollisionPlane), and Isaac's foot-vs-ground pattern
        # (many Robot links × one ground filter) is supported. Inverting would
        # make terrain the primary, which lacks ContactReportAPI.
        return prim, filters
    if sec_entity == "robot":
        return prim, filters
    # RigidObject (or any non-robot scene entity): single body primary.
    return _isaac_prim_path(sec_entity, None), [robot_filter]


def contact_sensor(
    backend: str,
    name: str,
    entity: str = "robot",
    pattern: str | None = None,
    secondary: str | Sequence[str] | None = None,
    secondary_pattern: str | Sequence[str] | None = None,
    track_air_time: bool = True,
    history_length: int = 3,
    fields: Sequence[str] = ("found", "force"),
    reduce: str = "netforce",
):
    """Build a backend contact-sensor cfg stored as ``scene.sensors[name]``.

    Task YAML lives under ``sensors.<name>`` (``_target_: contact_sensor``).

    Args that pick **which bodies** report contact
    ----------------------------------------------
    entity:
        Scene key of the *primary* side (who the sensor is “on”). Usually
        ``robot``, an object key from ``objects:`` (e.g. ``object``), rarely
        something else. Isaac maps ``robot`` → prim ``Robot``.
    pattern:
        Optional body-name regex **within** ``entity``. Narrows which links
        of that entity are primary bodies.

        * ``None`` (default): ``robot`` → all child links (``.*``); rigid
          objects → the object root only.
        * e.g. ``base_link`` or ``arm_joint.*`` to restrict the primary set.

    Args that pick **who they may contact** (filters / secondary match)
    --------------------------------------------------------------------
    secondary:
        Partner scene key(s): ``terrain``, ``robot``, or an object key. A
        string is one partner; a list is several (Isaac only — each becomes a
        filter channel on dim ``M``). Omit for unfiltered contacts (all
        partners → ``net_forces_w`` / mjlab primary-only).
    secondary_pattern:
        Optional body-name regex (or full Isaac prim path for ``terrain``)
        for the partner side — **not** the same as ``pattern``.

        * Aligns by index when ``secondary`` is a list; one string applies to
          every partner.
        * For ``secondary: terrain``, default Isaac filter is the plane leaf
          ``/World/ground/terrain/GroundPlane/CollisionPlane``; set this to
          ``/World/ground/terrain/mesh`` for generator terrains.
        * For ``secondary: robot``, default is all robot links (``.*``).

    Examples::

        sensors:
          robot_ground:
            entity: robot          # primary = robot links
            secondary: terrain     # only contacts vs ground
          robot_object:
            entity: robot
            secondary: object      # only contacts vs the capsule
          # Rare: only base vs ground, custom terrain prim
          base_ground:
            entity: robot
            pattern: base_link
            secondary: terrain
            secondary_pattern: /World/ground/terrain/mesh

    Backend notes
    -------------
    **Isaac:** read filtered forces from ``force_matrix_w`` ``(N, B, M, 3)``,
    not ``net_forces_w``. ``entity: robot`` + ``secondary: object`` with
    default multi-link ``.*`` auto-inverts (object primary, ``Robot/.*``
    filters) → ``(N, 1, B_robot, 3)``. ``secondary: terrain`` keeps robot
    primary → ``(N, B_robot, 1, 3)``. Do not use ``secondary: [terrain,
    object]`` on one Isaac sensor; declare two sensors.

    **mjlab:** one secondary partner per sensor; keep ``entity: robot`` as
    written. Multiple partners → separate ``sensors:`` entries.
    """
    isaac_pattern, mjlab_pattern = _resolve_body_pattern(entity, pattern)
    secondaries = _normalize_secondaries(secondary, secondary_pattern)

    if backend == "isaaclab":
        from isaaclab.sensors import ContactSensorCfg

        prim_path, filter_paths = _isaac_primary_and_filters(
            entity, isaac_pattern, secondaries
        )
        return ContactSensorCfg(
            prim_path=prim_path,
            track_air_time=track_air_time,
            history_length=history_length,
            filter_prim_paths_expr=filter_paths,
        )

    if backend == "mjlab":
        from mjlab.sensor import ContactMatch, ContactSensorCfg

        if len(secondaries) > 1:
            raise ValueError(
                "mjlab contact_sensor supports one secondary partner per sensor. "
                f"Got {len(secondaries)} ({[e for e, _ in secondaries]}). "
                "Declare separate sensors.<name> entries instead."
            )

        if not secondaries:
            secondary_match = None
        else:
            sec_entity, sec_pattern = secondaries[0]
            if sec_entity == "terrain":
                secondary_match = ContactMatch(
                    mode="body",
                    pattern=sec_pattern or "terrain",
                    entity=None,
                )
            else:
                _, sec_mj = _resolve_body_pattern(sec_entity, sec_pattern)
                secondary_match = ContactMatch(
                    mode="body",
                    pattern=sec_mj,
                    entity=sec_entity,
                )

        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="body", pattern=mjlab_pattern, entity=entity),
            secondary=secondary_match,
            fields=tuple(fields),
            reduce=reduce,
            num_slots=1,
            track_air_time=track_air_time,
            history_length=history_length,
        )

    raise ValueError(f"Unknown backend: {backend}")


registry.register("sensor", "contact_sensor", contact_sensor)
