# Simulation benchmark

Compare isaac and mjlab rollout FPS and GPU memory on the flat locomotion tasks.

```bash
NUM_ENVS="1024 4096" ./benchmark/run.sh
```

Results land in `benchmark/runs/<hostname>-<date>/`, which is gitignored. `report.md` in that directory is the table. Regenerate it with:

```bash
python3 benchmark/summarize.py --run-dir benchmark/runs/<stamp>
```

Each case is 6 PPO iterations (`algo=ppo`, `algo.in_keys=[policy]`, `wandb.mode=disabled`, `seed=0`, headless). The summary uses the median of iterations 1–5. Iteration 0 includes Warp or Kit warmup.

| Variable | Default | Meaning |
|---|---|---|
| `NUM_ENVS` | `1024 4096` | One suite per value |
| `ITERS` | `6` | PPO iterations per case |
| `PPO_TRAIN_EVERY` | `32` | Must match `algo.train_every`; sets `total_frames` |
| `ISAAC_PROJECT` | `venv/isaac51` | Skipped when the directory is missing |
| `MJLAB_PROJECT` | `venv/mjlab` | Skipped when the directory is missing |
| `STAMP` / `OUT` | hostname and date | Run directory name or full path |

Cases: `G1/G1LocoFlat` with `g1_waist_unlocked` and `g1_mjlab`, `A2/A2LocoFlat`, `Spot/SpotLocoFlat`, on whichever of the two backends is installed.

`Sim` in the report is the `simulation` timer. That timer is the CUDA-graph launch. The first reward that reads body state waits for the graph, so a heavy contact step shows up on that reward rather than on `Sim`. GPU memory is the max `nvidia-smi` sample at 1 Hz.

Do not commit `benchmark/runs/`. The numbers belong to one machine, one driver, and one Warp build.
