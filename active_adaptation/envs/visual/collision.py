"""InteriorGS / SAGE-3D collision USD → trimesh (shared frame with 3DGS).

Layout matches ``aa_scenes``: scene dir contains ``3dgs*.ply`` and
``{scene_id}_collision.usd`` (Z-up, same world frame as the Gaussians).

Prefer ``aa_scenes.collision`` when installed; otherwise use ``pxr``
(Isaac Sim / ``usd-core``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

COLLISION_SUFFIX = "_collision.usd"


def resolve_collision_usd(path: Path) -> Path | None:
    """Resolve a scene directory or PLY path to ``*_collision.usd``, or ``None``."""
    path = Path(path).resolve()
    directory = path.parent if path.is_file() else path
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"*{COLLISION_SUFFIX}"))
    return matches[0] if matches else None


def load_collision_trimesh(
    path: Path,
    *,
    color: tuple[float, float, float] = (0.72, 0.74, 0.78),
) -> trimesh.Trimesh:
    """Load ``{scene_id}_collision.usd`` as one world-frame trimesh.

    Same convention as ``aa_scenes.collision.load_collision_trimesh``: bake each
    ``UsdGeom.Mesh`` local-to-world transform, concatenate, assign flat color.
    """
    try:
        from aa_scenes.collision import load_collision_trimesh as _aa_load
    except ImportError:
        _aa_load = None
    if _aa_load is not None:
        return _aa_load(path, color=color)

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise ImportError(
            "Loading InteriorGS collision USD requires aa-scenes (usd-core) "
            "or pxr (Isaac Sim)."
        ) from exc

    usd_path = resolve_collision_usd(Path(path))
    if usd_path is None:
        raise FileNotFoundError(
            f"No *{COLLISION_SUFFIX} next to {path} "
            "(expected InteriorGS scene layout; see aa-scenes README)."
        )

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")

    time = Usd.TimeCode.Default()
    parts: list[trimesh.Trimesh] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        if points is None or counts is None or indices is None:
            continue

        vertices = np.asarray(points, dtype=np.float64)
        face_counts = np.asarray(counts, dtype=np.int64)
        face_indices = np.asarray(indices, dtype=np.int64)
        faces = _triangulate_faces(face_counts, face_indices)
        if len(faces) == 0:
            continue

        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        xform = np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time),
            dtype=np.float64,
        ).T
        part.apply_transform(xform)
        parts.append(part)

    if not parts:
        raise ValueError(f"No Mesh prims found in {usd_path}")

    combined = trimesh.util.concatenate(parts)
    combined.merge_vertices()
    rgba = np.array([*color, 1.0], dtype=np.float64)
    combined.visual.face_colors = np.tile(rgba * 255.0, (len(combined.faces), 1)).astype(
        np.uint8
    )
    return combined


def _triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    faces: list[list[int]] = []
    offset = 0
    for count in face_counts:
        count = int(count)
        face = face_indices[offset : offset + count]
        offset += count
        if count == 3:
            faces.append(face.tolist())
        elif count > 3:
            for k in range(1, count - 1):
                faces.append([int(face[0]), int(face[k]), int(face[k + 1])])
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)
