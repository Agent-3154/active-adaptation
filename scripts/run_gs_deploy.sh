#!/bin/bash
# Full GS deploy with composite video recording:
#   - Third-person GS (1.5,1.5,1.5 offset, with waypoint overlay)
#   - First-person pitched RGB
#   - DA3 raw depth
# All merged into single composite video per 500-step segment.

python scripts/deploy_instinctlab_gs.py \
    --task cfg/task/G1GsDeploy.yaml \
    --num_envs 1 \
    --max_steps 5000 \
    --episode_len 500 \
    --max_speed 0.5 \
    --diag_video_dir output/diag \
    --diag_interval 500 \
    --headless
