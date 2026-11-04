# Dex-Track

The collaborator project is pinned as `projects/dex-track` (Git submodule).
Run `git submodule update --init projects/dex-track` after checkout.
Its README covers the isolated mjlab runtime, audited collision patches,
current task configuration and data-pipeline/publication status.

This branch adds free-hand adapter state, fixed support variants, explicit
convex-solver capacity and reset-time FK/observation refresh. Per-substep FK
refresh is configurable. Existing upstream Isaac fixes are preserved.
Project tests, checkpoints, experimental records and outputs are not bundled.
No dataset or historical training snapshot is modified by this integration.

AA core PPO remains identical to main. Dex-Track owns its PPO and named wrist/finger action routing in dex_track_learning.
