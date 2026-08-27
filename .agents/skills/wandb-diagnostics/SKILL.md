---
name: wandb-diagnostics
description: Read WandB run diagnostics (grad norms, losses, PPO KL/entropy, explained variance) to analyze numerical stability and training behavior for RL algorithms in this repo. Use when the user provides a WandB run URL or run id and asks for algorithm analysis from logs.
disable-model-invocation: true
---

# WandB Diagnostics Reader

## What this skill does

- Fetches a WandB run’s recent history for a set of known diagnostic metric keys.
- Detects non-finite values (NaN/Inf) in key stability metrics (notably `critic/grad_norm`, `actor/grad_norm`).
- Produces a compact human summary plus a JSON payload that an agent can interpret programmatically.

## Main entrypoint (helper script)

Run:

```bash
uv run python -m active_adaptation.utils.wandb_diagnostics \
  --run <wandb_run_spec> \
  --samples 4000
```

Where `<wandb_run_spec>` can be one of:
- `run:<entity>/<project>/<run_id>`
- a `https://wandb.ai/...` URL
- `<entity>/<project>/<run_id>`

The command prints:
1) a short “Issues:” list
2) “Summary JSON:” with per-metric min/max/last and non-finite detection
3) a cache path under `.cache/wandb-diagnostics/<run_id>.json` (gitignored; for agent follow-up)

Use `--no-cache` to skip writing, or `--out <path>` to override the default path.

**Agent follow-up:** read the cached JSON instead of re-fetching WandB:

```bash
cat .cache/wandb-diagnostics/<run_id>.json
```

## Step-by-step workflow

1. Ask the user for the WandB run URL or run spec.
2. If the user also knows the algorithm family, tell the agent what keys to expect:
   - SAC-like: `critic/grad_norm`, `critic/q_loss`, `critic/q_upper`, `actor/grad_norm`
   - PPO-like: `critic/explained_var`, `actor/approx_kl`, `actor/entropy`
3. Execute the helper command above (or use the user’s requested `--samples`).
4. Use the JSON payload to decide what went wrong:
   - `critic/grad_norm` non-finite → numerical instability in critic (AMP overflow, target explosion, or loss outliers)
   - negative `critic/explained_var` → value head not tracking returns
   - large `actor/approx_kl` → policy update too aggressive (PPO LR/clip/trust region)
5. Propose code/config mitigations grounded in this repo:
   - critic non-finite: try `algo.use_amp=false` and/or `algo.critic_loss=huber`
   - BAC/BEE instability: set `algo.bee_lambda=0` (turn off BEE mixing) or lower LR
   - if you already changed SAC targets: re-check critic loss precision (fp32)

## Notes / caveats

- This tool reads WandB via `wandb.Api()`. It requires that the environment can access WandB and that credentials are available (typically via `wandb login`).
- Metric keys vary across configs. If the run doesn’t log one of the expected metrics, it will appear as empty in the JSON summary.
- If NaNs are intermittent, per-step detection may only flag the sampled history window controlled by `--samples`.

## Suggested output to return to the user

- 1 paragraph: main finding (e.g. “NaNs appear intermittently in `critic/grad_norm` after enabling BEE.”)
- 3–5 bullet points: most likely causes + concrete config-level fixes to try next.

