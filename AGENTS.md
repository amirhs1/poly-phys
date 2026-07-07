# AGENTS.md — PolyPhys

This is the repository-level operating guide for Codex. Keep it concise and mirror
`CLAUDE.md` as closely as possible. If the two files diverge, follow the stricter
approval or safety rule.

**Living document — last structural review: 2026-07-07.** Amir is actively restructuring
PolyPhys over the next few weeks: folders, module boundaries, and the packaging approach
are all expected to change. Treat the "Project" and "Packaging" sections below as a
snapshot, not gospel:

- Before relying on a specific path, module name, or command stated here, check it against the live repo (`ls`, `grep`, `pyproject.toml`). If they've diverged, the repo is right and this file is wrong — fix the file, don't silently work around the staleness.
- If your task changes anything this file states as fact (folder layout, package name,
build backend, test command, dependency groups, branch model, lineage vocabulary),
update the relevant section of AGENTS.md in the same PR and say so in the PR body.
- Bump the date above whenever you edit this file for a structural reason.

## Project

PolyPhys is a Python package for managing and analyzing polymer-physics/molecular-dynamics
simulation data, focused on computational models of bacterial chromosome organization
(`amirhs1/poly-phys`, MIT license, solo-maintained by Amir Sadeghi).

- The package structure is evolving. Inspect the current repository before relying on module,
  submodule, or file names; do not maintain a complete module inventory in this file.
- Preserve the settled simulation-artifact lineage:
  `segment → whole → ensemble_long → ensemble → space`.
- Preserve the settled directory phase/stage vocabulary:
  `simsAll, simsCont, logs, trjs, probes, analysis, viz, galaxy, allInOne` ×
  `segment, wholeSim, ens, ensAvg, space, galaxy`.
- Never rename, flatten, or simplify those conventions without being explicitly asked.
  Any filename-parsing change requires matching parser tests in the same change.
- Versioning: semantic versioning; the version string lives only in `polyphys/__version__.py`
  (`pyproject.toml` declares `dynamic = ["version"]` and reads it via `attr =`). Don't
  hand-edit the version unless asked to cut a release.
- No PyPI release yet — distribution is GitHub source plus a Zenodo DOI. Don't add a
  PyPI-publish workflow, trusted-publishing config, or version-bump automation unless
  explicitly asked.

## Branching & delivery

- The only long-lived branch is `main` — there is currently no `develop` branch. (If this
  changes, update this file.)
- Never commit or push directly to `main`.
- Every short-lived branch must use one of these prefixes, which determines its PR label:
  `feature/*` → `enhancement`; `fix/*` or `hotfix/*` → `bug`; `refactor/*` or
  `chore/*` → `chore`; `deps/*` → `dependencies`; `docs/*` → `documentation`;
  `help/*` → `help wanted`; `question/*` → `question`; `release/*` → `release`.
  You may branch from `main` without asking, but never commit directly to `main`.
- When work is ready, stop before staging or delivery and follow the approval checkpoints
  below. Never infer approval from the original task or from approval at an earlier step.
- **Before staging:** show `git status --short` and the unstaged diff (`git diff`), summarize
  the changed files, and ask Amir to review and explicitly approve staging. Stage only the
  approved files or hunks.
- **Before committing:** show the staged diff (`git diff --cached`) and the proposed commit
  message, then ask for explicit approval to commit.
- **Before pushing or opening a PR:** show the branch/base, proposed PR title and body, and
  ask for explicit approval to push and create the PR. Do not merge the PR — Amir owns merges.
- Keep commits scoped to the task; don't bundle unrelated changes.
- **Commit trailer.** End every commit you author or co-author with this trailer, exactly,
  as the final block, preceded by a blank line:

      Co-authored-by: Codex <noreply@openai.com>

  Verify with `git log -1 --pretty=%B` before pushing; re-verify after any amend.
- **Titles.** Don't prefix commit messages or PR titles with a tool/agent tag — the
  identifier lives only in the trailer above.
- **Labels.** Every PR must have exactly one label, determined by its branch prefix.
  Use only `bug`, `chore`, `dependencies`, `documentation`, `enhancement`,
  `help wanted`, `question`, or `release`. Never create or apply another label.
  If the branch prefix and intended label do not match, stop and ask before opening the PR.

## Packaging

- Build backend: setuptools (`pyproject.toml`, `[build-system]`). Package discovery includes
  `polyphys*`, excludes `notebooks*`, `projects*`, `under_review*`.
- Optional dependency groups: `docs` (Sphinx + pydata theme + myst-parser), `dev` (pytest,
  pytest-cov, flake8, mypy, pandas-stubs, build), `notebooks` (ipykernel/ipython/jupyter).
  Install only what the task needs, e.g. `pip install -e .[dev]`.
- `polyphys.test_data` ships as package data (`tool.setuptools.package-data`) — new test
  fixtures should land under that path, or update the glob.
- CLI entry point: `polyphys = "polyphys.cli:main"`.
- Before touching `pyproject.toml` or the package layout, verify the package still builds:
  `python -m build` (from the `dev` extra) — sdist and wheel should succeed.

## Testing — ask before running

Before running any test, doctest, lint, type-check, coverage, build, docs-build,
benchmark, or other validation command, list the exact proposed commands and ask Amir:

> Would you like to run these checks yourself, or should I run them?

Do not run validation until he answers. His choice applies to the current task only unless
he explicitly says otherwise. If he runs the checks, ask for or record the results; do not
claim they passed without evidence.

CI (`.github/workflows/ci.yaml`) runs on pushes to `main`, pull requests targeting `main`,
and manual dispatch. Relevant local commands that mirror CI are:

    ```bash
    python -m pip install -e .[dev]

    # Lint
    flake8 polyphys

    # Type-check
    mypy polyphys/analyze polyphys/manage

    # Tests + coverage + doctest-modules + README doctest
    pytest polyphys \
    --cov=polyphys \
    --cov-report=term-missing \
    --cov-report=xml \
    --doctest-modules \
    --doctest-glob="README.md"

    # Build sdist and wheel
    python -m build
    ```

- `[tool.pytest.ini_options]` already sets `addopts = "--doctest-modules
  --doctest-glob=README.md"` and `testpaths = ["polyphys/tests"]`, so a bare `pytest` from
  the repo root also works. Use the explicit command above when reproducing the CI test job.
- `--doctest-modules` means **docstring examples are executable tests**. If you add or edit
  a docstring `Examples` block, include the relevant doctests in the proposed validation.
- Codecov upload needs `CODECOV_TOKEN`, which only exists in CI. Do not try to reproduce the
  upload locally; the terminal and XML coverage reports are enough.
- If the change touches `docs/source/`, propose:
  `pip install -e .[docs] && sphinx-build -b html docs/source docs/_build` (autodoc mocks
  `MDAnalysis`, `pyarrow`, `statsmodels`, so those being "missing" at build time is expected).
- **CI is the gate.** It runs the test matrix on Python 3.11–3.13. For docs/config-only
  changes, propose only the narrow relevant local checks and let CI run the full matrix
  unless Amir asks for broader local validation.

## Domain correctness (don't skip this)

This package computes physical quantities (densities, volume fractions, persistence length,
etc. — see `polyphys.analyze.measurer`) from MD trajectories. Correctness here is *scientific*
correctness, not just type/lint correctness:

- Any change to a measurement/statistics function must preserve physical units and existing
  numerical test fixtures in `polyphys/tests/`. If expected values must change, explain why
  in the PR body (e.g., "previous formula omitted a factor of 2π — see [source]").
- New analysis or statistics code operating on ensembles of correlated MD frames should use
  block-averaging or another autocorrelation-aware error estimate, not naive per-frame
  standard error — flag this explicitly in the PR body if you implement or change error bars.
- Prefer vectorized NumPy/pandas/MDAnalysis operations over explicit per-frame Python loops;
  trajectories here can have millions of frames/particles, so note the time/space complexity
  (Big-O) of any new core routine in its docstring or the PR body, especially anything
  iterating over particles or frames.
- If you invoke a physical model or statistical method by name (e.g., Flory scaling, Rouse
  model, block averaging), name the source (paper or textbook) in the docstring or PR body —
  don't state it as if self-evident.

## Security

`SECURITY.md` exists: this is a solo-maintained project with a private-disclosure-only
policy and a 90-day window. Don't file or suggest filing security issues publicly; if you
find something that looks like a real vulnerability, stop and flag it to Amir directly
instead of opening a public issue or PR about it.

## Repository audit mode

If asked to audit or review the repository's health (not just fix a specific task), switch
to evidence-based audit mode:

- Work read-only by default. Don't modify, delete, or publish anything; don't run
  release/publish/clean commands.
- If installing audit-only tools (`pip-audit`, `bandit`, `pip-licenses`), do it inside a
  disposable venv (`python -m venv .audit-venv`) — never add them to `pyproject.toml`.
- If a command fails, retry at most once for an obvious fix; on a second failure, record the
  exact error and move on rather than looping.
- Score across: installation/packaging (20), code quality/architecture (20), testing & CI
  (20), documentation (15), security & dependency hygiene (15), maintenance/governance (10).
  For each finding, separate what you verified directly from what you assumed.
- Don't guess external metrics (PyPI downloads, GitHub stars, OpenSSF Scorecard) — only
  report them if a successful API/CLI call actually returned them; otherwise mark "could not
  be verified" rather than scoring it down.
- Output a single Markdown report; don't open a PR for an audit unless also asked to fix
  something.

## Final response

- If the task changed anything this file documents as fact (paths, module names, the
  lineage/phase vocabulary, build backend, test commands, branch model), update AGENTS.md
  in the same PR and bump the "last structural review" date at the top — don't let it go
  stale silently.
- Before proposing new features or audit-style recommendations, check
  `gh issue list --state all` first — don't re-propose something already filed; reference
  the issue number instead.
- Summarize changed files, why each change was made, and which checks were run (and which
  were skipped, and why).
- If the task scope is complete, present the current diff for Amir's review and stop before
  staging unless the relevant approval checkpoint has already been completed. Proceed through
  staging, commit, push, and PR creation only with the separate approvals above. Never merge
  the PR, and never push to `main` directly.
