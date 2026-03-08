source /root/miniconda3/bin/activate gsloco

cd /cpfs/user/benqingwei/gsloco/IsaacLab/ && ./isaaclab.sh -i
cd /cpfs/user/benqingwei/gsloco/active-adaptation/ && pip install -e .
cd /cpfs/user/benqingwei/gsloco/gs-scene/ && pip install -e .
cd /cpfs/user/benqingwei/gsloco/sirius-wheel-learning/ && pip install -e .

pip install depth-anything-3

sync && sleep 5

# Pre-compile gsplat CUDA kernels before torchrun to avoid multi-process race
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA compiled OK')"

cd /cpfs/user/benqingwei/gsloco/active-adaptation/scripts/

export OMP_NUM_THREADS=4

export http_proxy="https://benqingwei:M1u1Nzw0MrAzBSD4RSGv6uFTqzPCTpYfDzOSip3tmEEPGU00HhKErL9JpJHH@aliyun-proxy.pjlab.org.cn:13128"
export https_proxy="https://benqingwei:M1u1Nzw0MrAzBSD4RSGv6uFTqzPCTpYfDzOSip3tmEEPGU00HhKErL9JpJHH@aliyun-proxy.pjlab.org.cn:13128"
export HTTP_PROXY="https://benqingwei:M1u1Nzw0MrAzBSD4RSGv6uFTqzPCTpYfDzOSip3tmEEPGU00HhKErL9JpJHH@aliyun-proxy.pjlab.org.cn:13128"
export HTTPS_PROXY="https://benqingwei:M1u1Nzw0MrAzBSD4RSGv6uFTqzPCTpYfDzOSip3tmEEPGU00HhKErL9JpJHH@aliyun-proxy.pjlab.org.cn:13128"

export WANDB_API_KEY=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


torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train_ppo.py \
  task=G1AilabRoomNavDA3 task.num_envs=128 \
  wandb.project=g1_room wandb.run_name=G1AilabRoomNavDA3 \
  algo=ppo_geoloco headless=True total_frames=10_000_000_000 \
  task.scene.point_lights=false task.command.vis_waypoints=false
