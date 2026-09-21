# Rollout tensor compilation

`task.rollout_compile` selects compiled regions. When omitted, `observations`, `rewards`, and `nan_guard` are enabled. Reset runs eagerly. An explicit list replaces these defaults. To disable all rollout compilation:

```text
+task.rollout_compile=[]
```

- `observations`: dense observation group calculations and concatenation; TensorDict assembly stays eager. Functional observation groups are unchanged. Mimic-Lite caches ring-buffer selection indices during update to avoid specializing on the changing Python head.
- `rewards`: per-group reward values, weighting, modifiers, EMA updates and total. Stats TensorDict writes stay eager. This is a separate path from the legacy reward `_compile_` option; use one or the other.
- `nan_guard`: finite-row reductions and exceptional-row sanitization. Mode `error`/`sanitize`/`off`, offender reporting, integer leaves, and termination flags retain their existing semantics.

Compilation uses fullgraph tensor regions and disables CUDA graphs to avoid aliases between reused rollout buffers and compiler-owned outputs. Noisy observation terms are compiled together with deterministic terms. Random operations use eager random fallback, but fusion may reorder draws: identical seeds need not produce elementwise-identical outputs. CUDA checks cover the original uniform component, uniform-radius isotropic position, and isotropic rotation distributions, including marginal moments, quantiles, and environment/horizon/temporal correlations. History update/reset sampling remains unchanged. Warm-up/compilation time must be excluded from throughput measurements. `TORCH_LOGS=recompiles,graph_breaks` helps verify that steady-state calls reuse graphs; fullgraph failures are surfaced rather than silently falling back.

These flags do not enable compilation of physics or PPO. Tracking transforms already have their own compile wrappers. The framework and extension changes must be deployed together.

Validation artifacts and single-GPU measurements for HDMI are under `tasks/hdmi-public-pcd8k-20260920/compile/` in the workspace. Do not infer a speedup from reduced kernel count alone; compare untraced warm rollout latency on the same GPU.
