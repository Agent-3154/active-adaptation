# Agent skills (active-adaptation)

Portable playbooks for coding agents. **Canonical location:** `.agents/skills/` (committed; discovered by Cursor and other agents that read this path).

## Available skills

| Skill | Path | Use when |
|-------|------|----------|
| On-policy (PPO) | [onpolicy-algorithms/SKILL.md](onpolicy-algorithms/SKILL.md) | `learning/ppo/`, `train_ppo.py`, GAE, sym aug, Muon |
| Off-policy (SAC) | [offpolicy-algorithms/SKILL.md](offpolicy-algorithms/SKILL.md) | `learning/offpolicy/`, `train_offpolicy.py`, replay, RLPD |
| WandB diagnostics | [wandb-diagnostics/SKILL.md](wandb-diagnostics/SKILL.md) | Debugging/analysis from WandB run history (grad norms, losses, KL/entropy, explained variance) |
| Environment / MDP | [environment-mdp/SKILL.md](environment-mdp/SKILL.md) | `envs/mdp/`, `cfg/task/` obs/reward/term/action/command/rand |

Each skill has a `reference.md` with file maps and diagrams.

Also see [TEACHME.md](../../active_adaptation/learning/TEACHME.md) (shared style) and [AGENTS.md](../../AGENTS.md) (repo-wide conventions).

---

## How to use

### Cursor

1. **Workspace root** must contain `.agents/skills/<name>/SKILL.md`.
   - Open **`active-adaptation/`** as the workspace (recommended).
   - Monorepo parent (`lab51/`): symlink or copy skills — see FAQ below.

2. **Invocation**
   - `/offpolicy-algorithms` or `/onpolicy-algorithms` in chat
   - `@.agents/skills/onpolicy-algorithms/SKILL.md`
   - “Follow the on-policy agent skill”

3. **Auto-discovery** — Cursor may match the skill `description` in YAML frontmatter; explicit invocation is more reliable.

### Other agents (Claude Code, Codex, Aider, …)

1. **AGENTS.md** at repo root points here.
2. Prompt: “Read `.agents/skills/offpolicy-algorithms/SKILL.md` and follow it.”
3. Attach `@` the `SKILL.md` file in your tool.

### Humans

Open `SKILL.md` directly or start from `active_adaptation/learning/TEACHME.md`.

---

## Layout

```
active-adaptation/
├── .agents/skills/           # canonical (commit this)
│   ├── README.md
│   ├── onpolicy-algorithms/
│   ├── offpolicy-algorithms/
│   ├── environment-mdp/
│   └── wandb-diagnostics/
└── AGENTS.md                 # links agents here
```

---

## Adding a skill

1. Create `.agents/skills/<skill-name>/SKILL.md` (YAML frontmatter: `name`, `description`).
2. Optional `reference.md`.
3. Add a row to the table above and a pointer in `AGENTS.md`.

---

## FAQ

**Workspace is a parent monorepo, not `active-adaptation/`?**  
Cursor only scans `<workspace-root>/.agents/skills/`. Options: open `active-adaptation/` as the workspace, or at the monorepo root:

```bash
ln -sfn active-adaptation/.agents/skills .agents/skills
```

**Are skills applied to every chat?**  
No — opt-in via `/skill-name`, @-mention, or description match.

**Why TEACHME.md and skills?**  
`TEACHME.md` = shared style; skills = focused playbooks (on-policy / off-policy / env).
