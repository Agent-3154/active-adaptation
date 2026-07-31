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
"""
Online imitation / distillation training (teacher frozen, student separate).

Intended usage::

    uv run scripts/train_imitation.py task=A2/A2LocoManip \\
        teacher=ppo_symaug teacher.checkpoint_path=... student=interprior

Main loop follows ``train_offpolicy.py``: one env step at a time, then
``student.step(td)``. The student owns its buffer and update interval.
"""
from __future__ import annotations

import torch
import hydra
import wandb
import logging
import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import OmegaConf, DictConfig
from hydra.conf import HydraConf, RunDir, JobConf
from hydra.core.config_store import ConfigStore

from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
from torchrl.envs import TransformedEnv, Compose, InitTracker, StepCounter
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase

import active_adaptation as aa
from active_adaptation.pipeline_io import (
    RUN_STATE_FILENAME,
    get_run_state_dir,
    write_run_state,
)
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.utils.wandb import parse_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


DEFAULTS = [
    {"task": "A2/A2LocoManip"},
    {"algo@teacher": "from_checkpoint"},
    {"algo@student": "interprior"},
    "_self_",
]


@dataclass
class IsaacAppConfig:
    headless: bool = "${..headless}"
    enable_cameras: bool = "${..eval_render}"


@dataclass
class WandbConfig:
    name: str = "${..exp_name}/${now:%m-%d}_${now:%H-%M}"
    job_type: str = "train"
    project: str = "${oc.select:task.project,active_adaptation}"
    mode: str = "online"
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    """Hydra root config for teacher→student imitation / distillation."""

    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)

    headless: bool = True
    exp_name: str = (
        "${oc.select:task.name,test}-"
        "${oc.select:student.name,none}"
    )
    backend: str = "isaac"
    device: str = "cuda"

    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    total_frames: int = 150_000_000

    eval_render: bool = False
    log_interval: int = 32
    checkpoint_interval: int = 400
    upload_interval: int = 3200

    seed: int = 42
    # Optional resume path for the *teacher* (student uses ``checkpoint_path``).
    teacher_checkpoint_path: Optional[str] = None
    # Optional resume path for the *student* (teacher uses ``teacher_checkpoint_path``).
    checkpoint_path: Optional[str] = None
    discard_unused_obs: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)


cs = ConfigStore.instance()
cs.store(
    name="train_imitation",
    node=TrainConfig(
        hydra=HydraConf(
            run=RunDir(
                dir=(
                    "./outputs_train/${now:%Y-%m-%d}/"
                    "${now:%H-%M-%S}-${task.name}-${student.name}"
                )
            ),
            job=JobConf(chdir=True),
        )
    ),
)


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def make_env_teacher_student(
    task_cfg: DictConfig,
    teacher_cfg: DictConfig,
    student_cfg: DictConfig,
    seed: int,
    headless: bool,
    device: str,
    discard_unused_obs: bool = True,
    teacher_checkpoint_path: str | None = None,
    student_checkpoint_path: str | None = None,
) -> tuple[TransformedEnv, TensorDictModuleBase, TensorDictModuleBase]:
    """Build env + frozen teacher + student (disjoint parameters)."""
    from termcolor import colored
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.helpers import _ensure_backend_env_imported
    import active_adaptation

    seed = seed + aa.get_local_rank()
    backend = active_adaptation.get_backend()

    teacher_checkpoint = parse_checkpoint(teacher_checkpoint_path)
    student_checkpoint = parse_checkpoint(student_checkpoint_path)

    _ensure_backend_env_imported(backend)
    if backend == "isaac":
        env_cls = _EnvBase.registry[task_cfg.get("env_class", "IsaacBackendEnv")]
        env_device = str(device)
    elif backend == "mujoco":
        env_cls = _EnvBase.registry[task_cfg.get("env_class", "MujocoBackendEnv")]
        task_cfg.num_envs = 1
        task_cfg.reward = {}
        env_device = "cpu"
    elif backend == "mjlab":
        env_cls = _EnvBase.registry[task_cfg.get("env_class", "MjlabBackendEnv")]
        env_device = str(device)
    elif backend == "motrix":
        env_cls = _EnvBase.registry[task_cfg.get("env_class", "MotrixBackendEnv")]
        env_device = "cpu"
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # --- Teacher (frozen expert; no shared params with student) ---
    from active_adaptation.utils.checkpoint_cfg import (
        is_from_checkpoint_algo,
        load_algo_cfg_from_local_pt,
    )
    if is_from_checkpoint_algo(teacher_cfg):
        teacher_checkpoint.update()
        teacher_cfg = load_algo_cfg_from_local_pt(teacher_checkpoint.get_path())
    teacher_cfg = hydra.utils.instantiate(teacher_cfg)
    student_cfg = hydra.utils.instantiate(student_cfg)

    policy_in_keys = set(teacher_cfg.in_keys + student_cfg.in_keys)
    if not policy_in_keys:
        raise ValueError(
            "Specify `in_keys` on teacher and/or student configs "
            "(e.g. `command`, `policy`)."
        )

    if discard_unused_obs:
        for obs_group_key in list(task_cfg.observation.keys()):
            if obs_group_key not in policy_in_keys and not str(obs_group_key).endswith("_"):
                task_cfg.observation.pop(obs_group_key)
                print(
                    colored(
                        f"Discard obs group {obs_group_key} as it is not used.",
                        "yellow",
                    )
                )

    base_env = env_cls(task_cfg, env_device, headless=headless)

    if teacher_checkpoint is not None:
        teacher_checkpoint.update()
        teacher_path = teacher_checkpoint.get_path()
        print(f"[Info]: Using teacher checkpoint from: {teacher_path}")
        teacher_state = torch.load(teacher_path, weights_only=False) if teacher_path else {}
    else:
        teacher_state = {}

    if student_checkpoint is not None:
        student_checkpoint.update()
        student_path = student_checkpoint.get_path()
        print(f"[Info]: Using student checkpoint from: {student_path}")
        student_state = torch.load(student_path, weights_only=False) if student_path else {}
    else:
        student_state = {}

    transform = Compose(InitTracker(), StepCounter())
    env = TransformedEnv(base_env, transform)
    env.set_seed(seed)

    teacher_cls = teacher_cfg.get_class()
    print(f"Creating teacher {teacher_cls} on device {device}")
    teacher = teacher_cls.from_env(teacher_cfg, env, device=device)
    if "policy" in teacher_state:
        print(colored("[Info]: Load teacher from checkpoint.", "green"))
        teacher.load_state_dict(teacher_state["policy"])
    teacher.eval()

    # --- Student ---
    student_cls = student_cfg.get_class()
    print(f"Creating student {student_cls} on device {device}")
    student = student_cls.from_env(student_cfg, env, device=device)
    if "policy" in student_state:
        print(colored("[Info]: Load student from checkpoint.", "green"))
        student.load_state_dict(student_state["policy"])

    if hasattr(student, "make_tensordict_primer"):
        primer = student.make_tensordict_primer()
        if primer is not None:
            print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
            transform.append(primer)
            env = TransformedEnv(env.base_env, transform)

    return env, teacher, student


def run(cfg: TrainConfig) -> dict[str, str]:
    """Distill a student from a frozen teacher; return checkpoint paths."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, "
        f"local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    wandb_run = None
    run_dir = None
    if aa.is_main_process():
        wandb_run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        wandb_run.config.update(OmegaConf.to_container(cfg))
        wandb_run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = wandb_run.name.split("-")[-1]
        wandb_run.name = f"{run_idx}-{default_run_name}"
        setproctitle(wandb_run.name)

        run_dir = Path(wandb_run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        wandb_run.save(str(cfg_save_path), policy="now")
        wandb_run.save(str(run_dir / "config.yaml"), policy="now")

    from active_adaptation.helpers import evaluate
    from active_adaptation.utils.helpers import EpisodeStats

    env, teacher, student = make_env_teacher_student(
        task_cfg=cfg.task,
        teacher_cfg=cfg.teacher,
        student_cfg=cfg.student,
        seed=cfg.seed,
        headless=cfg.headless,
        device=cfg.device,
        discard_unused_obs=cfg.discard_unused_obs,
        teacher_checkpoint_path=cfg.teacher_checkpoint_path,
        student_checkpoint_path=cfg.checkpoint_path,
    )

    total_iters = cfg.total_frames // (aa.get_world_size() * env.num_envs)
    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval
    max_episode_length = cfg.task.max_episode_length
    log_interval = cfg.log_interval
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        ckpt_path = Path(wandb_run.dir) / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": wandb_run.name, "id": wandb_run.id}
        state_dict["policy"] = policy.state_dict()
        torch.save(state_dict, ckpt_path)
        if upload_to_wandb:
            wandb_run.save(str(ckpt_path), policy="now", base_path=wandb_run.dir)

        latest_link = Path(wandb_run.dir) / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        logging.info(
            f"Saved checkpoint to {ckpt_path}"
            + (" (wandb)" if upload_to_wandb else "")
        )
        return str(ckpt_path)

    assert env.training

    ckpt_path = None
    carry = env.reset()
    # teacher and student may each be stateful, e.g., running RNNs
    carry_teacher, carry_student = carry.copy(), carry.copy()
    # 0: teacher, 1: student
    policy_selection = torch.zeros(env.num_envs, dtype=bool, device=env.device)
    
    env_frames = 0
    private_keys = None
    observation_keys = list(env.observation_spec.keys(True, True))

    student.on_stage_start("train", env)
    teacher_rollout = teacher.get_rollout_policy(mode="eval")
    student_rollout = student.get_rollout_policy(mode="train")

    if aa.is_main_process():
        progress = tqdm(range(total_iters), desc="imitation")
    else:
        progress = range(total_iters)

    last_log_episode_stats = 0

    for i in progress:
        if hasattr(student, "step_schedule"):
            student.step_schedule(i / max(total_iters, 1))
        
        # resample policy selection upon episode start
        if (is_init := carry["is_init"]).any(): 
            is_init_ids = is_init.reshape(-1).nonzero().squeeze(1)
            student_frac = student.student_frac # TODO: refactor to a more general way
            policy_selection[is_init_ids] = torch.rand(len(is_init_ids), device=env.device) < student_frac

        with torch.no_grad():
            with (
                set_exploration_type(ExplorationType.RANDOM),
                ScopedTimer("policy_inference"),
            ):
                carry_teacher = teacher_rollout(carry_teacher)
                carry_student = student_rollout(carry_student)
                teacher_action = carry_teacher["action"]
                carry["policy_selection"] = policy_selection.clone()
                carry["teacher_action"] = teacher_action
                carry["action"] = torch.where(
                    policy_selection.unsqueeze(1),
                    carry_student["action"],
                    teacher_action,
                )

            with ScopedTimer("env_step") as timer:
                td, carry = env.step_and_maybe_reset(carry)
                carry_teacher.update(carry)
                carry_student.update(carry)
                if not private_keys:
                    private_keys = [
                        key
                        for key in td.keys(True, True)
                        if isinstance(key, str) and key.startswith("_")
                    ]
                td = td.exclude(*private_keys)
                td["next"] = td["next"].exclude(*observation_keys)

        episode_stats.add(td)
        new_frames = td.numel()
        env_frames += new_frames
        train_info: dict = student.step(td)

        if aa.is_main_process() and (i % checkpoint_interval == 0):
            should_upload = i % upload_interval == 0
            checkpoint_name = (
                f"checkpoint_{i}" if should_upload else "checkpoint_temp"
            )
            ckpt_path = save(
                student, checkpoint_name, upload_to_wandb=should_upload
            )

        if aa.is_main_process() and ((i % log_interval == 0) or len(train_info) > 0):
            info = {**train_info}
            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                (1 / timer.last_time) * new_frames * aa.get_world_size()
            )
            if (
                i - last_log_episode_stats >= max_episode_length
                and len(episode_stats) > 0
            ):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
                last_log_episode_stats = i

            ScopedTimer.print_summary(clear=True, depth=3)
            print(
                OmegaConf.to_yaml(
                    {k: v for k, v in info.items() if isinstance(v, (float, int))}
                )
            )
            info.update(env.extra)
            info.update(env.stats_ema)
            wandb_run.log(info)
            print(f"Latest checkpoint: {ckpt_path}")

    run_state: dict[str, str] = {}
    if aa.is_main_process():
        ckpt_path = save(student, "checkpoint_final")
        policy_eval = student.get_rollout_policy("eval")
        info, trajs, stats = evaluate(
            env, policy_eval, render=cfg.eval_render, seed=cfg.seed
        )
        info["env_frames"] = env_frames
        wandb_run.log(info)
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
        run_state = {
            "checkpoint_path": ckpt_path,
            "run_dir": str(run_dir),
            "task": str(cfg.task.name),
            "student": str(cfg.student.name),
            "teacher": str(cfg.teacher.get("name", cfg.teacher.get("_target_", ""))),
        }
        run_state_path = write_run_state(run_state, run_dir / RUN_STATE_FILENAME)
        print(f"Wrote run state to {run_state_path}")
        pipeline_dir = get_run_state_dir()
        if pipeline_dir is not None and pipeline_dir.resolve() != run_dir.resolve():
            write_run_state(run_state, pipeline_dir / RUN_STATE_FILENAME)
    return run_state


@hydra.main(
    config_path=str(CONFIG_PATH), config_name="train_imitation", version_base=None
)
def main(cfg: TrainConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
