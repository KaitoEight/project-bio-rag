# skills/

Reusable, documented procedures for maintaining this project. Each
sub-folder is a self-contained skill with its own `SKILL.md` + helpers.

| Skill | Purpose |
|---|---|
| [`etl-textbook-images/`](etl-textbook-images/SKILL.md) | Image ETL for scanned Vietnamese SGK PDFs (v7 anchor-first detector) |

When adding a new skill:
1. Create `skills/<skill-name>/SKILL.md` with the standard frontmatter.
2. Add the row above.
3. If the skill should be auto-discovered by Claude Code, also create
   `.claude/skills/<skill-name>/SKILL.md` (symlink or copy).
