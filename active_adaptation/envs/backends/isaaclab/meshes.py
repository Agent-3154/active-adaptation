"""USD → body-local trimesh extraction for Isaac entities.

Mirrors ``simple_raycaster`` / Viser conventions:
``{body}/visuals`` or ``{body}/collisions`` under ``root_physx_view.prim_paths[0]``.
Meshes are in the body frame for use with ``body_link_pose_w``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh


def entity_body_prim_paths(entity, suffix: str) -> list[str]:
    """Build ``{body}/{suffix}`` prim paths (same rules as MultiMeshRaycasterV2).

    Isaac articulations often use a container root prim (e.g. ``.../Robot``) whose
    name is not ``body_names[0]``. In that case children live at
    ``{root}/{body_name}/{suffix}``. When the root prim *is* the first body, fall
    back to string-replace like V2.
    """
    template_path = entity.root_physx_view.prim_paths[0]
    root_prim_name = template_path.rstrip("/").split("/")[-1]
    if root_prim_name == entity.body_names[0]:
        return [
            template_path.replace(root_prim_name, body_name) + f"/{suffix}"
            for body_name in entity.body_names
        ]
    return [f"{template_path}/{body_name}/{suffix}" for body_name in entity.body_names]


def _empty_trimesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )


def load_entity_body_meshes(
    entity,
    *,
    suffixes: Sequence[str],
    require_all: bool = True,
) -> list[trimesh.Trimesh]:
    """Extract one body-local trimesh per ``entity.body_names`` entry.

    Args:
        entity: Isaac Lab ``Articulation`` or ``RigidObject``.
        suffixes: Prim name candidates under each body, tried in order
            (e.g. ``(\"visuals\",)`` or ``(\"collisions\", \"collision\")``).
        require_all: If True, missing / ambiguous prims raise. If False,
            missing bodies get an empty trimesh (keeps ``num_bodies`` alignment).

    Returns:
        List of length ``entity.num_bodies`` in ``body_names`` order.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from simple_raycaster.utils_usd import find_matching_prims, get_trimesh_from_prim
    except ImportError as e:
        raise ImportError(
            "Isaac entity mesh extraction requires Isaac Sim and "
            "simple-raycaster (utils_usd)."
        ) from e

    if not suffixes:
        raise ValueError("suffixes must be non-empty")

    stage = get_current_stage()
    # Prefer the first suffix that yields any match for path layout; per body
    # we still try each candidate.
    path_lists = [entity_body_prim_paths(entity, s) for s in suffixes]
    meshes: list[trimesh.Trimesh] = []

    for body_i, body_name in enumerate(entity.body_names):
        mesh: trimesh.Trimesh | None = None
        last_err: Exception | None = None
        tried: list[str] = []
        for paths in path_lists:
            path = paths[body_i]
            tried.append(path)
            prims = find_matching_prims(path, stage)
            if len(prims) == 0:
                continue
            if len(prims) != 1:
                raise ValueError(
                    f"Expected exactly one prim for body '{body_name}' at "
                    f"'{path}', found {len(prims)}."
                )
            try:
                mesh = get_trimesh_from_prim(prims[0])
            except ValueError as e:
                last_err = e
                continue
            break

        if mesh is None:
            if require_all:
                detail = f" (extract failed: {last_err})" if last_err is not None else ""
                raise ValueError(
                    f"No mesh prim for body '{body_name}' under any of {tried}{detail}"
                )
            meshes.append(_empty_trimesh())
        else:
            meshes.append(mesh)

    if len(meshes) != entity.num_bodies:
        raise ValueError(
            f"Extracted {len(meshes)} body meshes but entity has "
            f"{entity.num_bodies} bodies."
        )
    return meshes
