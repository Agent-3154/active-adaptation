from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistributedMainProcessTest(unittest.TestCase):
    def _is_main(self, *, local_rank: int, rank: int) -> bool:
        env = os.environ.copy()
        env.update(
            LOCAL_RANK=str(local_rank),
            RANK=str(rank),
            WORLD_SIZE="16",
        )
        result = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sys, active_adaptation as aa; "
                "sys.stdout.write(str(int(aa.is_main_process())))",
            ],
            cwd=ROOT,
            env=env,
            text=True,
        )
        return bool(int(result))

    def test_only_global_rank_zero_is_main(self) -> None:
        self.assertTrue(self._is_main(local_rank=0, rank=0))
        self.assertFalse(self._is_main(local_rank=0, rank=8))


if __name__ == "__main__":
    unittest.main()
