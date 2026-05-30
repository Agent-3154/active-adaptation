"""
Backend-agnostic helpers for resolving names to indices.

Isaac Lab and MuJoCo expose slightly different APIs for looking up bodies and
joints. These functions normalize on the articulation's simulation name lists
(`joint_names_simulation`, `body_names_simulation`) so call sites get indices
and ordering that match the asset, not an arbitrary backend layout.
"""

try:
    from isaaclab.utils.string import resolve_matching_names
except (ImportError, ModuleNotFoundError):
    from mjlab.utils.lab_api.string import resolve_matching_names


def _get_contact_sensor_primary_names(contact_sensor) -> list[str]:
    """Return primary body names across Isaac Lab and mjlab contact sensors."""
    primary_names = getattr(contact_sensor, "primary_names", None)
    if primary_names is not None:
        return list(primary_names)

    slots = getattr(contact_sensor, "_slots", None)
    if slots is not None:
        names: list[str] = []
        seen: set[str] = set()
        for slot in slots:
            name = getattr(slot, "primary_name", None)
            if name is None or name in seen:
                continue
            names.append(name)
            seen.add(name)
        if names:
            return names

    raise AttributeError(
        "Contact sensor does not expose primary_names or mjlab _slots.primary_name"
    )


def find_sensor_bodies(
    asset,
    contact_sensor,
    body_names: str | list[str]
) -> tuple[list[int], list[str]]:
    """
    Resolve body name patterns to indices in the contact sensor.

    Names are first resolved via :func:`find_bodies`, so they follow
    ``asset.cfg.body_names_simulation`` (not the contact sensor's internal body
    order). On some stacks the articulation lists bodies breadth-first while the
    contact sensor lists them depth-first, so indices can disagree for the same
    name; this function maps names to the sensor using ``preserve_order=True``
    when supported, otherwise by indexing into ``contact_sensor.primary_names``.

    Returns:
        ``body_ids``: indices into the contact sensor's body arrays, in the same
        order as ``body_names`` (simulation order).
        ``body_names``: resolved names, same semantics as :func:`find_bodies`.
    """
    _, body_names = find_bodies(asset, body_names)
    if hasattr(contact_sensor, "find_bodies"):
        # IsaacLab API
        body_ids = contact_sensor.find_bodies(
            body_names,
            preserve_order=True,
        )[0]
    else:
        # MjLab API
        names = _get_contact_sensor_primary_names(contact_sensor)
        body_ids = []
        for name in body_names:
            try:
                body_ids.append(names.index(name))
            except ValueError as e:
                raise ValueError(f"Body name {name} not found in {names}") from e
    return body_ids, body_names


def find_joints(asset, joint_names: str | list[str]) -> tuple[list[int], list[str]]:
    """
    Resolve joint name patterns to articulation joint indices.

    Unlike ``asset.find_joints`` / ``entity.find_joints``, returned indices and
    name order follow ``asset.cfg.joint_names_simulation`` (simulation order),
    which keeps MDP code aligned with tensors indexed by that list.
    """
    _, joint_names = resolve_matching_names(joint_names, asset.cfg.joint_names_simulation)
    joint_ids = [
        asset.joint_names.index(name) for name in joint_names
    ]
    return joint_ids, joint_names


def find_bodies(asset, body_names: str | list[str]) -> tuple[list[int], list[str]]:
    """
    Resolve body name patterns to articulation body indices.

    Unlike ``asset.find_bodies`` / ``entity.find_bodies``, returned indices and
    name order follow ``asset.cfg.body_names_simulation`` (simulation order),
    which keeps MDP code aligned with tensors indexed by that list.
    """
    _, body_names = resolve_matching_names(body_names, asset.cfg.body_names_simulation)
    body_ids = [
        asset.body_names.index(name) for name in body_names
    ]
    return body_ids, body_names
