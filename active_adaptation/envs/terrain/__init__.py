import importlib
import os
import glob
from pathlib import Path

import active_adaptation

# Get all Python files in current directory
current_dir = Path(__file__).parent
terrain_files = glob.glob(os.path.join(current_dir, "*.py"))

# Import TERRAINS from each file
TERRAINS_MUJOCO = {}

if active_adaptation.get_backend() == "isaac":
    from . import regular
    from . wrapper import BetterTerrainGenerator, BetterTerrainImporter
else:
    from active_adaptation.envs.mujoco import MjTerrainCfg
    path = Path(active_adaptation.__path__[0]) / "assets_mjcf" / "plane.xml"
    TERRAINS_MUJOCO["plane"] = MjTerrainCfg(mjcf_path=str(path))

