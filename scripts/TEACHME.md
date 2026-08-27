# Rollout collection & management

Collect policy rollouts for offline replay, relabeling, and BC / off-policy bootstrapping. Archives live under `scripts/rollout/`.

Shared I/O: `active_adaptation/rollout_io.py`

---

## `rollout.py` — collect transitions

Runs the trained policy in eval mode and stacks transitions into a single archive per run.

### Launch

```bash
conda activate lab51
cd active-adaptation

# defaults (see cfg/rollout.yaml)
python scripts/rollout.py task=A2LocoManip algo=ppo_symaug checkpoint_path=/path/to/ckpt

# common overrides
python scripts/rollout.py task=A2LocoManip algo=ppo_symaug \
  checkpoint_path=wandb://... \
  num_steps=1000 \
  store_transitions=true \
  run_critic=true
```

### Config highlights (`cfg/rollout.yaml`)

| Key | Meaning |
|-----|---------|
| `num_steps` | Horizon `T` stacked per env (default: task `max_episode_length`) |
| `store_transitions` | If true, keep full obs in `next/*`; if false, strip obs keys |
| `run_critic` | Run critic head during rollout (affects stored `state_value`, etc.) |
| `checkpoint_path` | Checkpoint to load (wandb URI or local path) |

### Output layout

```
scripts/rollout/<task>-<algo>/<timestamp>/
  rollout_<T>_<N>.pt      # stacked TensorDict, batch [T, num_envs]
  rollout_<T>_<N>.json    # metadata companion
```

- `<T>` = `num_steps`, `<N>` = number of parallel envs
- Edited / merged copies use suffix tags, e.g. `rollout_1000_64_edited.pt`, `rollout_1000_192_merged.pt`

### Metadata JSON

Written beside each `.pt` file:

- `policy_name` — algo name
- `episode_count` — completed episodes during collection
- `episode_stats` — mean per-episode stats, e.g. `stats/loco/return`
- `tensor_entries` — per-key `shape`, `dtype`, `size_bytes`
- `tensor_shapes` — shape-only view (legacy compat)

Transition keys (typical): `action`, `policy`, `command`, `reward`, `done`, `next/*`, etc. Exact keys depend on task config.

---

## `rollout_manager.py` — browse, edit, combine

Gradio UI for inspecting and post-processing archives without re-running sim.

### Launch

```bash
conda activate lab51
cd active-adaptation
pip install -e ".[rollout]"   # once, for gradio

python scripts/rollout_manager.py
python scripts/rollout_manager.py --port 7861
```

Open `http://127.0.0.1:7860`.

> **Alternative:** `rollout_manager_nicegui.py` — same archives, staged inline edits with Save / Save As and per-row delete buttons.

### UI workflow

1. **Rollouts table** — check rows to select
2. **One selected** — edit keys (rename / mark delete), preview & apply edits
3. **Two+ selected** — view common keys only, combine (same `T` required)

### Key editing

Single-selection keys table: `key | shape | new_key | delete`

- Fill `new_key` to rename; check `delete` to drop a tensor
- **Preview edits** → **Apply & Save edits**

Save modes:

| Mode | Effect |
|------|--------|
| In-place (backup) | Overwrite `.pt` + `.json`, keep `.pt.bak` |
| In-place (no backup) | Overwrite directly |
| New file | Write `rollout_<T>_<N>_<suffix>.pt` (default suffix: `edited`) |

Selection is preserved after save (re-selects new file on “New file”).

### Combine

Select 2+ rollouts with the same `T` and identical key sets / compatible shapes. Concatenates on the env dimension: `[T, N1]` + `[T, N2]` → `[T, N1+N2]`. Output: `rollout_<T>_<N_sum>_merged.pt`.

---

## Common workflows

### Hindsight relabeling (dense → sparse command)

1. Collect with dense task: `rollout.py task=A2LocoManip ...`
2. Open manager, select rollout
3. Rename `command_sparse` → `command`, delete old `command` and unused obs keys
4. Save As → use for sparse BC / SAC prior training

### Merge batches for more envs

Collect two runs with same `T` and keys → combine in manager → single `[T, N1+N2]` archive for `ReplayBuffer.from_rollout`.

### Load in Python

```python
from active_adaptation.rollout_io import load_rollout, load_metadata
from pathlib import Path

path = Path("scripts/rollout/.../rollout_1000_64.pt")
payload = load_rollout(path)
stacked = payload["stacked"]          # TensorDict [T, N]
meta = load_metadata(path.with_suffix(".json"))
print(meta["episode_count"], meta["episode_stats"])
```

```python
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
buffer = ReplayBuffer.from_rollout(path)
```
