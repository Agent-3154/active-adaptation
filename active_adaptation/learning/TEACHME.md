# Guideline (for human) and Instruction (for agents) for Implementing a New Algorithm

This is a research-oriented codebase. We prefer single-file implementations that are conceptually clean and resembles the pseudocode as much as possible. We use `ppo` and `ppo_symaug` as templates (although they are not perfect).

The general style guidelines are:

- Try not to extract methods or fuctions unless they are low-level math primitives and can be reused widely.
- In most cases, do not use inheritance for the algorithm classes. For example, even if implementing a new PPO-style algorithm, do not inherit from `PPOBase` or `PPOPolicy`.
- Use type hints and docstrings to describe the code.
- Add informative diagnostics when possible, e.g., explained variance. Add explanatory comments when necessary.
- TODO: more style guidelines.
- Start with network architectures and hyperparameters used in the original paper/codebase (when available) when reproducing a new algorithm. Then add the common techniques in our codebase, e.g., Mish activation, ortho initialization, and Muon optimizer as we go.
- For agents: leave comments if uncertain about the implementation.

This doc is WIP. Let's try implementing a new algorithm step-by-step and refine the guidelines as we go.

## Notes after adding FPO++

- Keep a dedicated algorithm file under `learning/ppo/` and register a dataclass config via `ConfigStore` in the same file (for Hydra `algo=<name>` compatibility).
- For Hydra dataclass configs, avoid `Literal[...]` annotations on fields; use `str`/primitive types in config schema and validate allowed values at runtime.
- Reuse the existing rollout/training pipeline shape first (`train_op`, `_update`, `compute_advantage`, `get_rollout_policy`), then add algorithm-specific terms in `_update`.
- For PPO-style algorithms with multiple trust-region variants, prefer an explicit config switch (e.g., `trust_region_mode in {ppo, spo, aspo}`) rather than scattering objective branches.
- For flow-matching policies, rollout must emit all training-side ratio bookkeeping explicitly (e.g., `initial_cfm_loss`, `cfm_loss_eps`, `cfm_loss_t`, `x1_pred`) so `train_op` can stay purely on replayed tensors.
- If the ratio is CFM-based, compute it from `old_cfm_loss - new_cfm_loss` using the *same* `(eps, t)` samples stored during rollout, rather than introducing Gaussian log-prob surrogates.
- Add stabilization knobs to config with safe defaults (e.g., advantage clamp, log-ratio clamp); keep them optional and easy to disable.
- Add diagnostics tied to new behaviors (e.g., `actor/approx_kl`, `actor/clamp_ratio`, `critic/explained_var`, feature effective rank) so new algorithm behavior is visible in logs.
- If symmetry augmentation is used, keep it a config flag and ensure metrics still work when it is disabled.