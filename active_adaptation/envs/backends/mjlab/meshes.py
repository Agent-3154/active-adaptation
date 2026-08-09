"""MuJoCo → body-local trimesh extraction for mjlab entities.

Role split matches mjlab ``variants._classify_geom_role``:
visual = ``contype == 0`` and ``conaffinity == 0``; everything else is collision.

Meshes are in the body frame for use with ``body_link_pose_w`` (same contract as
Isaac ``get_visual_meshes`` / ``get_collision_meshes``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

GeomRole = Literal["visual", "collision"]


def _empty_trimesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )


def _is_visual_geom(mj_model, geom_id: int) -> bool:
    return (
        int(mj_model.geom_contype[geom_id]) == 0
        and int(mj_model.geom_conaffinity[geom_id]) == 0
    )


def _body_geom_ids(mj_model, body_id: int, *, role: GeomRole) -> list[int]:
    from mujoco import mjtGeom

    want_visual = role == "visual"
    out: list[int] = []
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[gid]) != body_id:
            continue
        gt = int(mj_model.geom_type[gid])
        # Skip non-renderable / unbounded geoms on robot bodies.
        if gt in (int(mjtGeom.mjGEOM_PLANE), int(mjtGeom.mjGEOM_HFIELD)):
            continue
        is_visual = _is_visual_geom(mj_model, gid)
        if is_visual == want_visual:
            out.append(gid)
    return out


def load_entity_body_meshes(
    entity,
    mj_model,
    *,
    role: GeomRole,
    require_all: bool = True,
) -> list[trimesh.Trimesh]:
    """Extract one body-local trimesh per ``entity.body_names`` entry.

    Args:
        entity: mjlab ``Entity`` (must be initialized; uses ``indexing.body_ids``).
        mj_model: compiled ``mujoco.MjModel`` from the simulation.
        role: ``\"visual\"`` or ``\"collision\"`` (contype/conaffinity).
        require_all: If True, bodies with no matching geoms raise. If False,
            they get an empty trimesh (keeps ``num_bodies`` alignment).
    """
    from mjviser.conversions import merge_geoms

    if not hasattr(entity, "indexing"):
        raise RuntimeError(
            f"Entity {getattr(entity, 'name', entity)!r} is not initialized; "
            "cannot resolve body_ids for mesh extraction."
        )

    body_ids = entity.indexing.body_ids.detach().cpu().tolist()
    if len(body_ids) != entity.num_bodies:
        raise ValueError(
            f"indexing.body_ids length {len(body_ids)} != num_bodies {entity.num_bodies}"
        )

    meshes: list[trimesh.Trimesh] = []
    for body_i, body_id in enumerate(body_ids):
        body_name = entity.body_names[body_i]
        geom_ids = _body_geom_ids(mj_model, int(body_id), role=role)
        if not geom_ids:
            if require_all:
                raise ValueError(
                    f"No {role} geoms for body '{body_name}' (mj body id {body_id})"
                )
            meshes.append(_empty_trimesh())
            continue
        meshes.append(merge_geoms(mj_model, geom_ids))

    if len(meshes) != entity.num_bodies:
        raise ValueError(
            f"Extracted {len(meshes)} body meshes but entity has "
            f"{entity.num_bodies} bodies."
        )
    return meshes
