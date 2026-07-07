# AGENTS.md — PolyPhys

Repository-level operating guide for Codex. It applies throughout the repository unless a
more specific instruction file applies. Keep it concise and mirror `CLAUDE.md` except for
agent-specific wording and the commit trailer. If the guides diverge, follow the stricter
approval or safety rule.

**Living document — last structural review: 2026-07-07.** PolyPhys is being actively
restructured; folder layout, module boundaries, and packaging details may change.

* Verify paths, modules, commands, branches, and packaging facts against the live repository
  (`ls`, `grep`, `pyproject.toml`, and relevant workflow files) before relying on this guide.
* When the repository contradicts this guide, the repository is authoritative. Update both
  guides in the same change instead of silently working around stale text.
* Bump the review date when a structural fact in either guide changes.

## Project invariants

PolyPhys is a solo-maintained, MIT-licensed Python package for polymer-physics and
molecular-dynamics simulation-data management and analysis, focused on bacterial chromosome
organization (`amirhs1/poly-phys`).

* Inspect the repository instead of maintaining a full module inventory here.
* Preserve the artifact lineage: `segment → whole → ensemble_long → ensemble → space`.
* Preserve the phase/stage vocabulary:
  `simsAll, simsCont, logs, trjs, probes, analysis, viz, galaxy, allInOne` ×
  `segment, wholeSim, ens, ensAvg, space, galaxy`.
* Never rename, flatten, or simplify those conventions unless Amir explicitly asks.
  Filename-parsing changes require matching parser tests in the same change.
* Versioning is semantic. The version lives only in `polyphys/__version__.py`;
  `pyproject.toml` reads it dynamically through `attr`. Change it only for a requested release.
* There is no PyPI release. Distribution is GitHub source plus a Zenodo DOI. Do not add PyPI
  publishing, trusted publishing, or version-bump automation unless asked.

## Local Conda environment and dependencies

The canonical local development environment is the named Conda environment `polylab_air`.

* Refer to it only by name. Never write or rely on an absolute Conda prefix or interpreter
  path such as `/opt/.../envs/polylab_air/bin/python` in tracked files.

* Run every Python-dependent project command through:

  ```txt
  conda run --no-capture-output -n polylab_air -- <command>
  ```

* Prefer module invocations:
  `conda run --no-capture-output -n polylab_air -- python -m <module>`.

* Do not run project commands with bare `python`, `pip`, `pytest`, `flake8`, `mypy`, or
  `sphinx-build`, even if the shell appears activated. Never use bare `pip`.

* Never install into Conda `base`, a system interpreter, another environment, or a
  user/global site. Never use `sudo` or `pip install --user`.

* Before any environment-changing command, including `pip install`, show the exact command
  and obtain explicit approval.

* Install only dependencies declared in `pyproject.toml` unless Amir explicitly approves a
  new dependency and its dependency group.

* Do not create, recreate, update, remove, clone, export, or otherwise administer the Conda
  environment unless Amir asks and approves the exact command.

* If `conda` or `polylab_air` is unavailable, stop and report it; never fall back silently.

* Non-Python commands such as `git`, `gh`, `ls`, and `grep` do not need Conda.

## Branching and delivery

* The only long-lived branch is `main`; there is no `develop` branch. Never commit or push
  directly to `main`.

* Branch prefix → single required PR label:
  `feature/*` → `enhancement`; `fix/*` or `hotfix/*` → `bug`;
  `refactor/*` or `chore/*` → `chore`; `deps/*` → `dependencies`;
  `docs/*` → `documentation`; `help/*` → `help wanted`;
  `question/*` → `question`; `release/*` → `release`.

* You may branch from `main` without asking. Keep branches and commits scoped to the task.

* Never infer delivery approval from the original task or an earlier checkpoint.

* Before staging: show `git status --short` and `git diff`, summarize changes, and request
  explicit approval. Stage only approved files or hunks.

* Before committing: show `git diff --cached` and the proposed message, then request
  explicit approval.

* Before pushing or opening a PR: show branch/base and the proposed PR title/body, then
  request explicit approval. Do not merge; Amir owns merges.

* Do not prefix commit or PR titles with an agent/tool tag.

* End every commit you author or co-author with this exact final block after a blank line:

  ```txt
  Co-authored-by: Codex <noreply@openai.com>
  ```

* Verify the trailer with `git log -1 --pretty=%B` before pushing and after any amend.

* If branch prefix and intended label disagree, stop and ask. Never create or apply labels
  outside the approved list.

## Packaging

* Build backend: setuptools in `pyproject.toml`.
* Discovery includes `polyphys*`; it excludes `notebooks*`, `projects*`, and `under_review*`.
* Optional groups: `docs`, `dev`, and `notebooks`. Install only what the task needs:
  `conda run --no-capture-output -n polylab_air -- python -m pip install -e '.[dev]'`.
* `polyphys.test_data` ships as package data. Put fixtures there or deliberately update the
  package-data glob.
* CLI entry point: `polyphys = "polyphys.cli:main"`.
* After changing `pyproject.toml` or package layout, propose:
  `conda run --no-capture-output -n polylab_air -- python -m build`.

## Validation — ask before running

Before any test, doctest, lint, type-check, coverage, build, docs build, benchmark, or other
validation command, list the exact proposed commands and ask:

> Would you like to run these checks yourself, or should I run them?

Do not run validation until Amir answers. The choice applies only to the current task unless
he says otherwise. If Amir runs checks, record his results; never claim success without
evidence.

CI (`.github/workflows/ci.yaml`) runs on pushes to `main`, PRs targeting `main`, and manual
dispatch. Commands mirroring CI are:

```bash
# Installation changes the environment and requires separate approval.
conda run --no-capture-output -n polylab_air -- python -m pip install -e '.[dev]'

conda run --no-capture-output -n polylab_air -- python -m flake8 polyphys

conda run --no-capture-output -n polylab_air -- \
  python -m mypy polyphys/analyze polyphys/manage

conda run --no-capture-output -n polylab_air -- \
  python -m pytest polyphys --cov=polyphys --cov-report=term-missing \
  --cov-report=xml --doctest-modules --doctest-glob='README.md'

conda run --no-capture-output -n polylab_air -- python -m build
```

* Pytest configuration already sets doctest options and `testpaths = ["polyphys/tests"]`.
  A minimal invocation is
  `conda run --no-capture-output -n polylab_air -- python -m pytest`.
* Docstring `Examples` blocks are executable tests; include relevant doctests when they change.
* Codecov upload needs `CODECOV_TOKEN`, available only in CI. Do not reproduce the upload
  locally; terminal and XML reports are sufficient.
* For `docs/source/` changes, propose the separately approved install
  `python -m pip install -e '.[docs]'` and then
  `python -m sphinx -b html docs/source docs/_build`, both through the required `conda run`
  prefix. Missing `MDAnalysis`, `pyarrow`, and `statsmodels` is expected because docs mock them.
* CI is the gate and runs Python 3.11–3.13. For docs/config-only changes, propose narrow
  local checks and let CI run the matrix unless Amir asks otherwise.

## Scientific and performance correctness

* Measurement/statistics changes must preserve physical units and numerical fixtures. If
  expected values change, explain the scientific reason and source in the PR.
* For correlated MD frames, use block averaging or another autocorrelation-aware uncertainty
  estimate, not naive per-frame standard errors.
* Prefer vectorized NumPy, pandas, and MDAnalysis operations over explicit Python loops.
* State time and space complexity for every new or materially changed core routine,
  especially code scaling with frames or particles.
* For a physical model, statistical method, algorithm, or nontrivial formula, cite a
  verifiable paper, textbook, standard reference, or official library documentation in the
  docstring, documentation, or PR body as appropriate.

## Security and audit mode

* `SECURITY.md` requires private disclosure with a 90-day window. Do not open a public issue
  or PR for a suspected vulnerability; stop and alert Amir privately.
* Audits are read-only by default. Do not modify, delete, publish, release, or clean unless
  separately asked.
* Install audit-only tools (`pip-audit`, `bandit`, `pip-licenses`) only in a disposable
  `.audit-venv`, never in `polylab_air` or `pyproject.toml`.
* Retry a failed audit command once only for an obvious correction; then record the error.
* Audit weights: installation/packaging 20, code quality/architecture 20, testing/CI 20,
  documentation 15, security/dependency hygiene 15, maintenance/governance 10.
* Separate verified, likely-but-untested, and unassessed findings. Never invent metrics.
* Before proposing features or audit recommendations, run `gh issue list --state all` and
  reference existing issues rather than duplicating them.

## Completion

* If a task changes a documented fact, update both guides in the same change and bump the
  structural-review date when appropriate.
* In the final response, summarize changed files, why each changed, checks run, checks
  skipped, and why.
* When implementation is complete, present the diff and stop before staging unless that
  checkpoint has explicit approval.
