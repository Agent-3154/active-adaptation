# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Imitation / distillation policies (teacher frozen, student separate)."""

import os
import importlib
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore

# Side-effect: register ``algo=*`` configs so we can mirror them under ``teacher``.
import active_adaptation.learning.ppo  # noqa: F401
from active_adaptation.learning.ppo.ppo_symaug import PPOConfig as SymaugCfg
from active_adaptation.learning.ppo.ppo_teacher_student import PPOTSCfg

dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if file.endswith(".py") and not file.startswith("_"):
        importlib.import_module(f".{file[:-3]}", __package__)


@dataclass
class TeacherSymaugCfg(SymaugCfg):
    """``teacher=ppo_symaug`` — PPO symaug as a frozen expert."""

    _target_: str = "active_adaptation.learning.imitation.TeacherSymaugCfg"
    checkpoint_path: Optional[str] = None


@dataclass
class TeacherTSCfg(PPOTSCfg):
    """``teacher=ppo_teacher`` — privileged teacher–student recipe as expert."""

    _target_: str = "active_adaptation.learning.imitation.TeacherTSCfg"
    checkpoint_path: Optional[str] = None


def _register_teacher_aliases() -> None:
    """Expose common PPO algos as ``teacher=<name>`` with ``checkpoint_path``."""
    cs = ConfigStore.instance()
    cs.store(name="ppo_symaug", node=TeacherSymaugCfg, group="teacher")
    cs.store(name="ppo_teacher", node=TeacherTSCfg(stage="teacher"), group="teacher")


_register_teacher_aliases()
