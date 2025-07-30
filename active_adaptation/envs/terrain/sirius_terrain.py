from isaaclab.terrains import (
    TerrainImporterCfg,
    HfTerrainBaseCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshRailsTerrainCfg,
    MeshStarTerrainCfg,
    SubTerrainBaseCfg,
    FlatPatchSamplingCfg
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG as ROUGH_HARD
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.utils import configclass
from dataclasses import MISSING
import numpy as np
import trimesh

import isaaclab.sim as sim_utils


def ramp_terrain(difficulty: float, cfg: "RampTerrainCfg"):
    height = cfg.height_range[0] + (cfg.height_range[1] - cfg.height_range[0]) * difficulty
    
    mesh = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], height))

    up = np.random.rand() > 0.5
    if up:
        # remove the bottom face
        bottom_faces = np.where(mesh.triangles_center[:, 2] <= -height / 2)
        mesh.faces = np.delete(mesh.faces, bottom_faces, axis=0)
        # bevel top edges
        top_vertices = mesh.vertices[mesh.vertices[:, 2] >= height / 2]
        top_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] >= height / 2] = top_vertices
        mesh.vertices[:, 2] += height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, height])
    else:
        # remove the top face
        top_faces = np.where(mesh.triangles_center[:, 2] >= height / 2)
        mesh.faces = np.delete(mesh.faces, top_faces, axis=0)
        # bevel bottom edges
        bottom_vertices = mesh.vertices[mesh.vertices[:, 2] <= -height / 2]
        bottom_vertices[:, 0] *= 0.5
        mesh.vertices[mesh.vertices[:, 2] <= -height / 2] = bottom_vertices
        mesh.vertices[:, 2] -= height / 2
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -height])
        # flip the normals for correct collision
        mesh.faces = np.fliplr(mesh.faces)
    # center the mesh
    mesh.vertices += np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    return [mesh], origin


@configclass
class RampTerrainCfg(SubTerrainBaseCfg):
    function = ramp_terrain
    height_range: tuple[float, float] = MISSING


def room_terrain(difficulty: float, cfg: "RoomTerrainCfg"):
    mesh_list, origin = flat_terrain(difficulty, cfg)
    wall_0 = trimesh.creation.box(extents=(4.0, 0.4, 1.0))
    wall_0.apply_translation(np.array([2.0, 0.2, 0.5]))
   
    wall_1 = wall_0.copy()
    wall_1.apply_transform(
        trimesh.transformations.transform_around(
            matrix=trimesh.transformations.rotation_matrix(
                angle=np.pi / 2,
                direction=np.array([0.0, 0.0, 1.0]),
            ),
            point=np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0]),
        )
    )
    mesh_list.append(wall_0)
    mesh_list.append(wall_1)
    return mesh_list, origin


@configclass
class RoomTerrainCfg(SubTerrainBaseCfg):
    function = room_terrain



def double_prism(difficulty: float, cfg):
    height_scale = cfg.height_range[0] + (cfg.height_range[1] - cfg.height_range[0]) * difficulty
    sink = np.random.uniform(cfg.sink_range[0], cfg.sink_range[1])
    sink = np.clip(sink, 0.0, height_scale)

    meshes = []
    prism_0 = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0., 0.], [0., 1.], [2., 0.]]),
        faces=np.array([[0, 1, 2]]),
        height=1,
    )
    # prism_0.apply_translation(np.array([-1., 0., 0.]))
    prism_0.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,
            direction=np.array([1, 0, 0])
        )
    )
    prism_0.apply_translation([-0.5, -0.5, 0.])
    meshes.append(prism_0)

    prism_1 = trimesh.creation.extrude_triangulation(
        vertices=np.array([[0., 0.], [0., -1.], [-2., 0.]]),
        faces=np.array([[0, 1, 2]]),
        height=1
    )
    # prism_1.apply_translation(np.array([1., 0., 0.]))
    prism_1.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=-np.pi / 2,
            direction=np.array([1, 0, 0])
        )
    )
    prism_1.apply_translation([0.5, 0.5, 0.])
    meshes.append(prism_1)

    box_0 = trimesh.creation.box(
        extents=(3.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, 0.5]),
    )
    meshes.append(box_0)
    box_1 = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([1.0, 1.0, 0.5]),
    )
    meshes.append(box_1)
    box_2 = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0),
        transform=trimesh.transformations.translation_matrix([-1.0, -1.0, 0.5]),
    )
    meshes.append(box_2)

    # Combine meshes
    mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.apply_translation([1.5, 1.5, -sink])
    mesh.apply_scale(np.array([cfg.size[0] / 3, cfg.size[1] / 3, height_scale]))
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, height_scale - sink])
    return [mesh], origin


@configclass
class DoublePrismCfg(SubTerrainBaseCfg):
    function = double_prism
    size = (8.0, 8.0)
    height_range = (1.0, 2.0)
    sink_range = (0.1, 0.2)


ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(9.0, 9.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        # "grid": MeshRandomGridTerrainCfg(
        #     proportion=0.20, 
        #     grid_width=0.45, 
        #     grid_height_range=(0.02, 0.05), 
        #     platform_width=2.0
        # ),
        "double_prism": DoublePrismCfg(
            proportion=0.20,
            size=(8.0, 8.0),
            height_range=(1.0, 2.0),
            sink_range=(0.1, 0.2),
        ),
        "rails": MeshRailsTerrainCfg(
            proportion=0.20,
            rail_height_range=(0.05, 0.25),
            rail_thickness_range=(0.2, 0.4),
            platform_width=2.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.20,
            num_bars=5,
            bar_width_range=(0.8, 1.0),
            bar_height_range=(1.0, 1.0),
            platform_width=4.0,
        ),
        "room": RoomTerrainCfg(
            proportion=0.20,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.2, 0.4),
            platform_width=4.0,
        ),
    },
)


PLANE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
        improve_patch_friction=True
    ),
)

ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MISSING,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)


TERRAINS = {
    "sirius_easy": ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_EASY),
    "sirius_hard": ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_HARD),
}


