# Reference: `active_adaptation.utils.wandb_diagnostics`

## CLI

```bash
uv run python -m active_adaptation.utils.wandb_diagnostics \
  --run <wandb_run_spec> \
  --samples 4000 \
  --keys critic/grad_norm,critic/q_loss,critic/q_upper,actor/approx_kl
```

By default writes `.cache/wandb-diagnostics/<run_id>.json` (gitignored). Flags: `--no-cache`, `--out <path>`.

## Run spec formats

- `run:<entity>/<project>/<run_id>`
- `https://wandb.ai/<entity>/<project>/runs/<run_id>`
- `<entity>/<project>/<run_id>`

## Output JSON shape (high level)

```json
{
  "run_id": "...",
  "run_name": "...",
  "run_url": "...",
  "state": "...",
  "metrics": {
    "critic/grad_norm": {
      "summary": {"min": ..., "max": ..., "last": ...},
      "nonfinite": {"nonfinite_count": ..., "nonfinite_steps_sample": [...]}
    }
  },
  "issues": ["..."]
}
```

