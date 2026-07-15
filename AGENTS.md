# AGENTS.md — PolyPhys operating contract

This file contains stable, repository-wide instructions for coding agents. Keep it concise. The live repository is authoritative for structural and tooling facts.

## Repository purpose

PolyPhys is a solo-maintained, MIT-licensed Python package for polymer-physics and molecular-dynamics simulation-data management and analysis, focused on bacterial chromosome organization (`amirhs1/poly-phys`).

## Establish the current state first

At the start of a task:

1. Inspect the current branch and worktree state when a local checkout is available.
2. Read the focused issue or PR, relevant source, tests, documentation, `pyproject.toml`, and applicable workflows.
3. Identify the smallest coherent change and the checks that can verify it.
4. State material assumptions and keep the work limited to the authorized scope.

The live worktree and current GitHub metadata take precedence over stale prose. Report material conflicts instead of silently choosing one source.

## Instruction scope and sources of truth

- These instructions apply repository-wide unless a more specific instruction file applies to the files being changed.
- Put shared directory-specific guidance in a nested `AGENTS.md`. Add a sibling `CLAUDE.md` containing `@AGENTS.md`.
- Keep nested guidance additive. When it replaces a root rule, name the replaced rule explicitly.
- Use parallel agents only for independent tasks. Do not let multiple agents edit the same files concurrently; use separate branches or worktrees.
- Canonical sources are `README.md`, `pyproject.toml`, `.github/workflows/`, `docs/source/`, `polyphys/__version__.py`, and `SECURITY.md`.

## Project invariants

- Preserve the artifact lineage: `segment → whole → ensemble_long → ensemble → space`.
- Preserve the phase/stage vocabulary:
  `simsAll, simsCont, logs, trjs, probes, analysis, viz, galaxy, allInOne` ×
  `segment, wholeSim, ens, ensAvg, space, galaxy`.
- Do not rename, flatten, or simplify those conventions unless Amir explicitly requests it. Filename-parsing changes require matching parser tests.
- Use semantic versioning. Change the version only for an explicitly requested release.
- Distribution is GitHub source plus a Zenodo DOI. Do not add PyPI publishing, trusted publishing, or automated version bumps unless requested.

## Default work sequence

1. **Understand:** inspect the issue, code, tests, docs, configuration, and CI.
2. **Plan:** identify affected modules, scientific assumptions, tests, docs, and compatibility implications.
3. **Implement:** make the smallest coherent task-scoped change.
4. **Verify:** run the narrowest relevant checks, then broaden when warranted.
5. **Self-review:** inspect the complete branch-versus-base diff for correctness, unrelated changes, secrets, generated artifacts, and stale documentation.
6. **Deliver:** commit, push, and open or update a draft PR when the task authorizes implementation.
7. **Report:** distinguish completed, verified, unverified, and deferred work.

Ask a focused question only for a material product, scientific, scope, release, destructive-action, or high-risk decision that cannot be resolved from the repository.

## Python environments and dependencies

On Amir's local workstation, the canonical environment is the named Conda environment `polylab_air`.

- Refer to it only by name. Never put an absolute Conda prefix or interpreter path in tracked files.
- On Amir's workstation, run Python-dependent commands through:

  ```text
  conda run --no-capture-output -n polylab_air -- <command>
  ```

- Prefer module invocations such as `python -m pytest`, `python -m flake8`, and `python -m build`.
- On Amir's workstation, do not use bare Python tooling and do not mutate any environment without explicit approval.
- In CI, Codex cloud, Claude Code web, containers, and other disposable environments, use the runtime-supplied environment. Declared dependencies may be installed during setup when platform policy permits.
- Never install into Conda `base`, a system interpreter, or a user/global site. Never use `sudo` or `pip install --user`.
- Do not add, remove, or upgrade project dependencies, optional groups, or lockfiles without explicit approval.
- If required tooling is unavailable, report the blocked checks. Do not silently switch environments or change project configuration.
- Approved audit-only tools must use a disposable environment such as `.audit-venv`; never add them to `polylab_air` or `pyproject.toml`.

## Python, documentation, and test standards

- Preserve the Python range declared in `pyproject.toml`.
- Use NumPy-style docstrings with Sphinx/reStructuredText markup. Preserve Sphinx/reST roles and directives inside docstrings.
- Keep type annotations authoritative. Document semantics, units, array shapes, accepted ranges, side effects, and scientific assumptions.
- Every new or materially changed public function, method, and class must include a meaningful `Examples` section with small, deterministic examples.
- Every new or materially changed test function or method must have a concise docstring stating the behavior, invariant, or regression it protects.
- Add or update focused tests for every behavior change and bug fix. Do not rewrite unrelated legacy tests merely to satisfy newer style rules.
- Update the relevant `docs/source/` page or README for every new or materially changed user-facing feature.
- Treat doctest examples as executable tests and run them when their output or underlying behavior changes.

## Validation and review

Run existing non-destructive checks without separate permission. Start narrow and broaden when warranted. On Amir's workstation, prefix each Python command below with the required Conda wrapper.

```bash
# Environment-changing on Amir's workstation; approval required there.
python -m pip install -e '.[dev]'

python -m flake8 polyphys
python -m mypy polyphys/analyze polyphys/manage
python -m pytest polyphys README.md   --cov=polyphys --cov-report=term-missing --cov-report=xml   --doctest-modules --doctest-glob='README.md'
python -m build
```

- For a narrow first pass, target the changed test module.
- For documentation changes, run relevant doctests and, when declared docs dependencies are available, run `python -m sphinx -b html docs/source docs/_build`.
- CI is authoritative for the full supported Python matrix. Local checks may be narrower; report their scope accurately.
- Never claim a command, build, test, CI run, benchmark, visual check, or audit passed unless its successful result was observed.
- Review the final diff and generated files before delivery.

## Scientific and performance correctness

- Preserve physical units, array-shape contracts, numerical meaning, and established scientific assumptions.
- Explain changed expected values and their scientific basis in the PR.
- For correlated molecular-dynamics frames, use block averaging or another autocorrelation-aware uncertainty estimate instead of naive per-frame standard errors.
- Prefer vectorized NumPy, pandas, and MDAnalysis operations when they improve performance without obscuring correctness.
- Document time and space complexity for every new or materially changed core routine whose cost scales with frames, particles, or dataset size.
- Cite a verifiable paper, textbook, standard, or official library document for physical models, statistical methods, algorithms, and nontrivial formulas.

## Git and draft PR policy

The only long-lived branch is `main`; there is no `develop` branch.

- Never commit or push directly to `main`.
- Use one focused non-`main` branch per meaningful task where practical.
- Branch prefix determines the single routine PR label:
  - `feature/*` → `enhancement`
  - `fix/*` or `hotfix/*` → `bug`
  - `refactor/*` or `chore/*` → `chore`
  - `deps/*` → `dependencies`
  - `docs/*` → `documentation`
  - `help/*` → `help wanted`
  - `question/*` → `question`
  - `release/*` → `release`
- When the current task explicitly authorizes a focused implementation, that authorization covers creating the branch, editing code/tests/docs, making coherent commits, pushing the focused branch, opening or updating a draft PR, and applying the matching routine label. Do not ask again for each routine step.
- Before the first push, inspect `git status --short` and the complete branch-versus-base diff; check for unrelated files, generated artifacts, secrets, private data, and accidental deletions; and run relevant checks.
- After maintainer review begins, do not amend published commits, rebase, or force-push unless requested or explicitly approved.
- Do not add agent/tool prefixes to commit or PR titles. Use the tool's configured native attribution; do not hard-code model names or add duplicate attribution trailers.
- If the branch prefix and intended label disagree, stop and ask.
- The maintainer alone may mark a PR ready, approve, merge, enable auto-merge, publish a release, or alter repository protections.

## High-risk changes

Obtain explicit approval before pushing changes involving:

- workflow permissions, privileged triggers, repository settings, or branch protection
- new or upgraded third-party dependencies
- licensing or attribution policy
- breaking public API changes not already authorized by the task
- destructive migrations, broad deletion, or irreversible data changes
- release versions, tags, publication, or Zenodo release metadata
- secrets, credentials, private data, or sensitive datasets
- force-pushing after review begins

## Packaging, CI, and security

- Treat `pyproject.toml` as authoritative for packaging, dependency groups, package discovery, package data, and entry points.
- After packaging, package-data, entry-point, or version-loading changes, run `python -m build` in the applicable environment.
- For GitHub Actions, use least-privilege permissions, avoid privileged triggers that execute untrusted code, keep commands locally reproducible, and inspect failed job logs before proposing a fix.
- Never expose, log, commit, or paste credentials, tokens, private keys, or sensitive datasets.
- Follow `SECURITY.md`: report suspected vulnerabilities privately and do not open a public issue or PR containing exploit details.
- Security audits are read-only by default. Separate verified, likely-but-untested, and unassessed findings.
- Before creating a tracked issue, inspect open and closed issues when network access is available; otherwise state that duplication was not checked.

## Completion report

Report:

- what changed and why
- files changed
- tests and exact outcomes
- checks not run and why
- draft PR, branch, and label updates
- scientific or compatibility assumptions
- remaining risks and what Amir should review before marking the PR ready
