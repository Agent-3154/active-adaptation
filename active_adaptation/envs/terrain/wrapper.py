import torch
import numpy as np
import warnings
from isaaclab.terrains import (
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainImporter,
    TerrainImporterCfg,
)
import isaaclab.sim as sim_utils

class BetterTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg, device)
        self.num_cols = self.cfg.num_cols
        self.num_rows = self.cfg.num_rows
        warnings.warn("Hacking TerrainGenerator. Check IsaacLab regularly for updates and compatibility.")
        
    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        # different from the original TerrainGenerator, we store the sub-terrain type mapping
        self.sub_terrain_types = torch.zeros(
            self.cfg.num_rows * self.cfg.num_cols,
            dtype=torch.int32,
            device=self.device
        )
        self.sub_terrain_type_mapping = {key: i for i, key in enumerate(self.cfg.sub_terrains.keys())}

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])
            self.sub_terrain_types[index] = sub_index

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        self.sub_terrain_types = torch.zeros(
            self.cfg.num_rows * self.cfg.num_cols,
            dtype=torch.int32,
            device=self.device
        )
        self.sub_terrain_type_mapping = {key: i for i, key in enumerate(self.cfg.sub_terrains.keys())}

        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])
                self.sub_terrain_types[sub_row * self.cfg.num_cols + sub_col] = sub_indices[sub_col]

    def get_terrain_origin_id(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Return the terrain cell index (flat) for each world position using grid layout.

        Assumes axis-aligned grid: cell (row, col) has origin at
        origin_00 + (row * size[0], col * size[1], 0). Positions are clamped to valid range.

        Args:
            pos_w: World positions, shape (N, 3).

        Returns:
            Flat cell indices in [0, num_rows*num_cols), shape (N,), dtype long, same device as pos_w.
        """
        device = pos_w.device
        size_x, size_y = self.cfg.size[0], self.cfg.size[1]
        origin_x = - 0.5 * size_x * self.cfg.num_rows
        origin_y = - 0.5 * size_y * self.cfg.num_cols
        # Cell in grid: row = floor((pos_x - origin_00_x) / size_x), col = floor((pos_y - origin_00_y) / size_y)
        delta_x = pos_w[:, 0] - origin_x
        delta_y = pos_w[:, 1] - origin_y
        row = (delta_x / size_x).long().clamp(0, self.num_rows - 1)
        col = (delta_y / size_y).long().clamp(0, self.num_cols - 1)
        return (row * self.num_cols + col).to(device)

    def get_sub_terrain_type(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Return the sub-terrain type index for each world position.

        Uses grid layout and cell sizes to compute the containing cell (no distance computation).

        Args:
            pos_w: World positions, shape (N, 3).

        Returns:
            Sub-terrain type indices, shape (N,), dtype int32, on the same device as pos_w.
        """
        cell_ids = self.get_terrain_origin_id(pos_w)
        return self.sub_terrain_types[cell_ids]


class BetterTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        warnings.warn("Hacking TerrainImporter. Check IsaacLab regularly for updates and compatibility.")
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()
        self.terrain_generator = None
        
        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
            self.terrain_generator = terrain_generator
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
    