# CLAUDE.md — PolyPhys

@AGENTS.md

## Shared contract

`AGENTS.md` is the canonical operating contract for every coding agent. Do not
restate shared rules here; add only Claude Code-specific behavior.

## Local and scoped instructions

- Read `CLAUDE.local.md` when present. Keep it gitignored and never commit it.
- Put genuinely path-specific Claude guidance in `.claude/rules/` so it loads
  only for matching files.
- Put recurring Claude-only procedures in `.claude/skills/` so they load on
  demand rather than expanding this file.

## Permissions and enforcement

Treat instruction files as behavioral guidance, not technical enforcement.
Obey Claude Code permissions, sandbox settings, hooks, and GitHub branch
protection. Never bypass a denied command or weaken a permission rule.

## Environment

In Claude Code web or another disposable remote environment, follow the remote
environment rules in `AGENTS.md`; do not require the workstation-only
`polylab_air` Conda environment.

## Git attribution

Use Claude Code's current `attribution` configuration. Do not rely on the
deprecated `includeCoAuthoredBy` setting, hard-code a model name, or add a
duplicate attribution block.

## Draft pull requests

For an authorized focused implementation, Claude may commit, push the
non-`main` branch, and open or update a draft PR under the policy in
`AGENTS.md`. Leave the PR in draft state for Amir's review. Report any requested
PR metadata that the available GitHub integration could not set.
