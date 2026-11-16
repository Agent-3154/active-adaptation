import importlib
import os
import glob
from pathlib import Path

import active_adaptation

# Get all Python files in current directory
current_dir = Path(__file__).parent
terrain_files = glob.glob(os.path.join(current_dir, "*.py"))

if active_adaptation.get_backend() == "isaaclab":
    from . import regular
    from . wrapper import BetterTerrainGenerator, BetterTerrainImporter

