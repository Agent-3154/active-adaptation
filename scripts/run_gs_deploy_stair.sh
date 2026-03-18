#!/bin/bash
# Headless GS deploy on stair scene with composite video recording.

python scripts/deploy_instinctlab_gs.py \
    --task cfg/task/G1GsDeployStair.yaml \
    --num_envs 1 \
    --max_steps 5000 \
    --episode_len 500 \
    --max_speed 0.5 \
    --da3_model depth-anything/DA3-LARGE \
    --diag_video_dir output/diag_stair \
    --diag_interval 500 \
    --headless
