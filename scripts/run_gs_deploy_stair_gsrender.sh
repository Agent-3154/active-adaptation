#!/bin/bash
# Headless GS deploy on stair scene using direct GS depth rendering (no DA3).

python scripts/deploy_instinctlab_gs.py \
    --task cfg/task/G1GsDeployStair.yaml \
    --num_envs 1 \
    --max_steps 5000 \
    --episode_len 500 \
    --max_speed 0.5 \
    --depth_mode gs_render \
    --diag_video_dir output/diag_stair_gsrender \
    --diag_interval 500 \
    --headless
