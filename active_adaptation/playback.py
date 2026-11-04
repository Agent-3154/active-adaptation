"""Shared interactive playback used by AA and project entry points."""

import itertools
import time

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.utils.timerfd import Timer


def play_policy(
    env,
    policy,
    video_path,
    *,
    record_video=False,
    exploration_type=ExplorationType.MODE,
    realtime=True,
):
    """Play indefinitely, retaining normal termination/reset and bounded stats."""
    stats_keys = [
        k for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval").to(env.device)
    env.base_env.eval()
    carry = env.reset()
    assert not env.base_env.training
    timer = Timer(env.step_dt) if realtime else None
    last_print_time = time.perf_counter()
    last_print_step = -1

    with env.get_recorder(video_path, enabled=record_video) as rec, \
        torch.inference_mode(), set_exploration_type(ExplorationType(exploration_type)):
        try:
            for i in itertools.count():
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)
                if record_video:
                    rec.add_frame()
                if len(episode_stats) >= env.num_envs:
                    print("Step", i)
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        print(k, torch.mean(v).item())
                now = time.perf_counter()
                elapsed = now - last_print_time
                if elapsed >= 2.0:
                    n_steps = i - last_print_step
                    sps = n_steps / elapsed
                    print(f"step {i} | {sps:.1f}x{env.num_envs}={sps*env.num_envs:.1f} env steps/s")
                    last_print_time = now
                    last_print_step = i
                if timer is not None:
                    timer.sleep()
        except KeyboardInterrupt:
            print(f"Interrupted by user, video saved to: {video_path}" if record_video else "Interrupted by user.")
