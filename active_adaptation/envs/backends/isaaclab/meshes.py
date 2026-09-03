"""USD → body-local trimesh / geom-part extraction for Isaac entities.

Mirrors ``simple_raycaster`` / Viser conventions:
``{body}/visuals`` or ``{body}/collisions`` under ``root_physx_view.prim_paths[0]``.
Meshes are in the body frame for use with ``body_link_pose_w``.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple

import trimesh


def entity_body_prim_paths(entity, suffix: str) -> list[str]:
    """Build ``{body}/{suffix}`` prim paths for Isaac articulation / rigid bodies.

    Isaac articulations often use a container root prim (e.g. ``.../Robot``) whose
    name is not ``body_names[0]``. In that case children live at
    ``{root}/{body_name}/{suffix}``. When the root prim *is* the first body, fall
    back to string-replace like V2.

    Empty ``suffix`` means the body prim itself (useful for USD meshes hanging
    directly under the link, or Isaac shape assets whose geom lives under
    ``geometry/`` but can also be discovered from the body root).
    """
    template_path = entity.root_physx_view.prim_paths[0]
    root_prim_name = template_path.rstrip("/").split("/")[-1]
    trail = f"/{suffix}" if suffix else ""
    if root_prim_name == entity.body_names[0]:
        return [
            template_path.replace(root_prim_name, body_name) + trail
            for body_name in entity.body_names
        ]
    return [f"{template_path}/{body_name}{trail}" for body_name in entity.body_names]


def _resolve_body_prims(entity, suffixes: Sequence[str], stage):
    from simple_raycaster.utils_usd import find_matching_prims

    if not suffixes:
        raise ValueError("suffixes must be non-empty")
    path_lists = [entity_body_prim_paths(entity, s) for s in suffixes]
    for body_i, body_name in enumerate(entity.body_names):
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
            yield body_i, body_name, prims[0], tried
            break
        else:
            yield body_i, body_name, None, tried


def load_entity_body_meshes(
    entity,
    *,
    suffixes: Sequence[str],
    require_all: bool = True,
) -> Tuple[list[int], list[str], list[trimesh.Trimesh]]:
    """Extract body-local trimeshes for bodies that have geometry.

    Tessellates USD Mesh + primitives (Cube/Sphere/Cylinder/Capsule/Cone).
    Returns ``(body_indices, body_names, meshes)`` for non-empty bodies.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from simple_raycaster.utils_usd import get_trimesh_from_prim
    except ImportError as e:
        raise ImportError(
            "Isaac entity mesh extraction requires Isaac Sim and "
            "simple-raycaster (utils_usd)."
        ) from e

    stage = get_current_stage()
    body_indices: list[int] = []
    body_names: list[str] = []
    meshes: list[trimesh.Trimesh] = []

    for body_i, body_name, prim, tried in _resolve_body_prims(entity, suffixes, stage):
        mesh: trimesh.Trimesh | None = None
        last_err: Exception | None = None
        if prim is not None:
            try:
                mesh = get_trimesh_from_prim(prim)
            except ValueError as e:
                last_err = e
                mesh = None
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            if require_all:
                detail = (
                    f" (extract failed: {last_err})" if last_err is not None else ""
                )
                raise ValueError(
                    f"No mesh prim for body '{body_name}' under any of {tried}{detail}"
                )
            warnings.warn(
                f"Body '{body_name}': no mesh geometry under {tried}"
                + (f" ({last_err})" if last_err is not None else "")
                + "; skipping body.",
                stacklevel=2,
            )
            continue
        body_indices.append(body_i)
        body_names.append(body_name)
        meshes.append(mesh)
    return body_indices, body_names, meshes


def load_entity_body_geom_parts(
    entity,
    *,
    suffixes: Sequence[str],
    require_all: bool = True,
):
    """Extract per-geom parts for Viser (native primitives when possible).

    Returns ``(body_indices, body_names, parts_per_body)`` where each
    ``parts_per_body[i]`` is a ``list[UsdGeomPart]``.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from simple_raycaster.utils_usd import get_geom_parts_from_prim
    except ImportError as e:
        raise ImportError(
            "Isaac entity mesh extraction requires Isaac Sim and "
            "simple-raycaster (utils_usd)."
        ) from e

    stage = get_current_stage()
    body_indices: list[int] = []
    body_names: list[str] = []
    parts_per_body: list = []

    for body_i, body_name, prim, tried in _resolve_body_prims(entity, suffixes, stage):
        parts = []
        last_err: Exception | None = None
        if prim is not None:
            try:
                parts = get_geom_parts_from_prim(prim)
            except ValueError as e:
                last_err = e
                parts = []
        if not parts:
            if require_all:
                detail = (
                    f" (extract failed: {last_err})" if last_err is not None else ""
                )
                raise ValueError(
                    f"No mesh prim for body '{body_name}' under any of {tried}{detail}"
                )
            warnings.warn(
                f"Body '{body_name}': no mesh geometry under {tried}"
                + (f" ({last_err})" if last_err is not None else "")
                + "; skipping body.",
                stacklevel=2,
            )
            continue
        body_indices.append(body_i)
        body_names.append(body_name)
        parts_per_body.append(parts)
    return body_indices, body_names, parts_per_body


# Default suffixes for Viser / MeshRegistry visuals:
# robots → ``visuals``; Isaac shape RigidObjects / custom USD → body root
# (subtree includes ``geometry/``; parts stay in the body frame for
# ``body_link_pose_w``).
VISUAL_GEOM_SUFFIXES: tuple[str, ...] = ("visuals", "")


def load_prim_geom_parts(
    prim_path: str,
    *,
    child_suffixes: Sequence[str] = ("", "geometry", "visuals"),
    require_all: bool = False,
):
    """Extract geom parts under a USD prim (e.g. ``AssetBaseCfg`` extras).

    Tries ``prim_path`` first so parts are expressed in the same frame as
    ``XformPrimView.get_world_poses()`` on that prim. Falls back to
    ``geometry`` / ``visuals`` children when needed.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from simple_raycaster.utils_usd import find_matching_prims, get_geom_parts_from_prim
    except ImportError as e:
        raise ImportError(
            "Isaac prim mesh extraction requires Isaac Sim and "
            "simple-raycaster (utils_usd)."
        ) from e

    stage = get_current_stage()
    tried: list[str] = []
    last_err: Exception | None = None
    for suffix in child_suffixes:
        path = f"{prim_path}/{suffix}" if suffix else prim_path
        tried.append(path)
        prims = find_matching_prims(path, stage)
        if len(prims) == 0:
            continue
        if len(prims) != 1:
            raise ValueError(
                f"Expected exactly one prim at '{path}', found {len(prims)}."
            )
        try:
            parts = get_geom_parts_from_prim(prims[0])
        except ValueError as e:
            last_err = e
            parts = []
        if parts:
            return parts

    if require_all:
        detail = f" (extract failed: {last_err})" if last_err is not None else ""
        raise ValueError(f"No mesh geometry under any of {tried}{detail}")
    warnings.warn(
        f"No mesh geometry under {tried}"
        + (f" ({last_err})" if last_err is not None else "")
        + "; skipping prim.",
        stacklevel=2,
    )
    return []


def load_prim_trimesh(
    prim_path: str,
    *,
    child_suffixes: Sequence[str] = ("", "geometry", "visuals"),
    require_all: bool = True,
) -> trimesh.Trimesh:
    """Combined body-local trimesh under a USD prim (AssetBase extras)."""
    parts = load_prim_geom_parts(
        prim_path, child_suffixes=child_suffixes, require_all=require_all
    )
    if not parts:
        raise ValueError(f"No mesh geometry under {prim_path!r}")
    meshes = [p.mesh for p in parts]
    combined: trimesh.Trimesh = (
        meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    )
    if len(combined.vertices) == 0 or len(combined.faces) == 0:
        raise ValueError(f"Empty mesh under {prim_path!r}")
    combined.merge_vertices()
    return combined
