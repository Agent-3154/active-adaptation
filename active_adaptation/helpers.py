import torch
import hydra
import numpy as np
import time
import os
import datetime
import importlib
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as imageio

from typing import Union
from termcolor import colored
from omegaconf import DictConfig

from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, StepCounter
from tensordict import TensorDictBase

import active_adaptation
from active_adaptation.utils.wandb import parse_checkpoint
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.envs import _EnvBase


class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


def _ensure_backend_env_imported(backend: str):
    backend_modules = {
        "isaac": "active_adaptation.envs.backends.isaac",
        "mujoco": "active_adaptation.envs.backends.mujoco",
        "mjlab": "active_adaptation.envs.backends.mjlab",
        "motrix": "active_adaptation.envs.backends.motrix",
    }
    module_name = backend_modules.get(backend)
    if module_name is None:
        raise ValueError(f"Unknown backend: {backend}")
    importlib.import_module(module_name)


def make_env_policy(
    task_cfg: DictConfig,
    algo_cfg: DictConfig,
    seed: int,
    headless: bool,
    device: str,
    discard_unused_obs: bool = True,
    checkpoint_path: str | None = None
) -> tuple[Union[TransformedEnv, _EnvBase], torch.nn.Module]:
    """Build env and policy, optionally restoring from `checkpoint_path`."""

    seed = seed + active_adaptation.get_local_rank()
    # Select the appropriate backend-specific environment class
    backend = active_adaptation.get_backend()

    # Parse checkpoint in parallel with environment creation.
    with ThreadPoolExecutor(max_workers=1) as executor:
        checkpoint_future = executor.submit(parse_checkpoint, checkpoint_path)
        
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
        
        policy_in_keys = algo_cfg.get("in_keys", None)
        if policy_in_keys is None:
            raise ValueError("Specify `in_keys` (e.g., `policy`, `priv`) in `cfg.algo`.")

        if discard_unused_obs:
            def should_discard(key: str) -> bool:
                return (
                    key not in policy_in_keys
                    and not key.endswith("_")
                )
            for obs_group_key in list(task_cfg.observation.keys()):
                if should_discard(obs_group_key):
                    task_cfg.observation.pop(obs_group_key)
                    print(colored(f"Discard obs group {obs_group_key} as it is not used.", "yellow"))

        base_env = env_cls(task_cfg, env_device, headless=headless)
        checkpoint = checkpoint_future.result()

    if checkpoint is not None:
        checkpoint.update()
    checkpoint_path = checkpoint.get_path() if checkpoint else None

    print(f"[Info]: Using checkpoint from: {checkpoint_path}")
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}
    
    transform = Compose(InitTracker(), StepCounter())

    env = TransformedEnv(base_env, transform)
    env.set_seed(seed)
    
    # setup policy
    policy_cls = hydra.utils.get_class(algo_cfg._target_)
    print(f"Creating policy {policy_cls} on device {device}")
    policy = policy_cls.from_env(algo_cfg, env, device=device)
    
    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])
    
    if hasattr(policy, "make_tensordict_primer"):
        primer = policy.make_tensordict_primer()
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
        env = TransformedEnv(env.base_env, transform)

    return env, policy


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm

@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    render=False,
    keys=[("next", "stats")],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """
    keys = set(keys)
    keys.add(("next", "done"))

    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs = []
    frames = []

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    max_episode_length = env.cfg.max_episode_length
    with ScopedTimer("rollout"), set_exploration_type(exploration_type):
        for i in tqdm(range(max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(*keys, strict=False).cpu())
            if render:
                frames.append(env.render("rgb_array"))
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    compute_std_for = ["return", "survival"]
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        if k in compute_std_for:
            info[key + "_std"] = torch.std(v.float()).item()

    # log video
    if len(frames):
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_path = os.path.join(os.path.dirname(__file__), f"recording-{time_str}.mp4")
        with imageio.get_writer(video_path, fps=int(1 / env.step_dt), codec="h264") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        frames.clear()

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats
