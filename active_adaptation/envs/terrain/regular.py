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
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshRailsTerrainCfg,
    height_field
)
from dataclasses import MISSING

import isaaclab.sim as sim_utils


ROUGH_HARD = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.2, 0.5),
            platform_width=4.0,
        )
    },
)


ROUGH_MEDIUM = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
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
        # "random_rough_easy": HfRandomUniformTerrainCfg(
        #     proportion=0.15,
        #     noise_range=(0.0, 0.06),
        #     noise_step=0.02,
        #     border_width=0.5
        # ),
        # "boxes": MeshRandomGridTerrainCfg(
        #     proportion=0.15,
        #     grid_width=0.45, 
        #     grid_height_range=(0.02, 0.05), 
        #     platform_width=2.0
        # ),
        # "box": MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.20,
        #     object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     platform_width=2.0
        # ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.15, 0.25),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

ROUGH_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
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
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.06),
            noise_step=0.01,
            border_width=0.5
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.20, 
            grid_width=0.45, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)




STAIRS = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
            border_width=0.5
        ),
        # "box": MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.20,
        #     object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=36, height=0.15, size=(0.6, 0.6), max_yx_angle=15),
        #     platform_width=2.0
        # ),
        # "rail": MeshRailsTerrainCfg(
        #     proportion=0.20,
        #     rail_thickness_range=(0.2, 0.3),
        #     rail_height_range=(0.2, 0.3),
        #     platform_width=2.5,
        # ),
        "pit": MeshPitTerrainCfg(
            proportion=0.20,
            pit_depth_range=(0.2, 0.4),
            platform_width=4.0,
        ),
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_b": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.20),
            step_width=0.50,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

STAIRS_TEST = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.20),
            step_width=0.35,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_b": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.10, 0.20),
            step_width=0.50,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)


SLOPES_AND_CURBS = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.1),
            step_width=0.40,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, 
            grid_width=0.60, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
    },
)

STAIRS_EASY = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "random_rough_easy": HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.04),
            noise_step=0.02,
            border_width=0.5
        ),
        "pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "pyramid_stairs_inv_a": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.40,
            platform_width=3.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

SLOPES_AND_BOXES = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.5, 
            grid_width=0.60, 
            grid_height_range=(0.02, 0.05), 
            platform_width=2.0
        ),
        "pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
        ),
        "pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.10, 0.20),
            platform_width=1.0,
            border_width=0.25
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

from active_adaptation.registry import Registry

registry = Registry.instance()
registry.register("terrain", "medium", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_MEDIUM))
registry.register("terrain", "medium_curriculum", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_MEDIUM.replace(curriculum=True)))
registry.register("terrain", "easy", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_EASY))
registry.register("terrain", "hard_curriculum", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_HARD))
registry.register("terrain", "plane", PLANE_TERRAIN_CFG)
registry.register("terrain", "stairs", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=STAIRS))
registry.register("terrain", "stairs_test", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=STAIRS_TEST))
registry.register("terrain", "stairs_easy", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=STAIRS_EASY))
