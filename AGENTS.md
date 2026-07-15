# AGENTS.md — PolyPhys

Shared repository instructions for coding agents. This is the single source of truth for Codex and Claude Code;
`CLAUDE.md` imports it and must not duplicate it.

**Living document — last structural review: 2026-07-14.** PolyPhys is actively being restructured. Verify structural
facts against the live repository before relying on this guide.

## Scope and working method

- These instructions apply repository-wide unless a more specific instruction file applies to the files being changed.
- Before editing, inspect the relevant source, tests, docs, `pyproject.toml`, and workflows. The live repository is
  authoritative when it conflicts with this file.
- Keep changes task-scoped. Do not perform unrelated cleanup, renaming, dependency changes, or refactoring.
- For nontrivial work: inspect first, state a concise plan, implement in small steps, run relevant checks, and review the
  final diff.
- Never report a command as passing unless its successful output was observed. State which checks were not run and why.
- Add durable conventions or recurring review feedback here. Bump the review date only when structural facts or the
  instruction architecture change.
- Put shared directory-specific guidance in a nested `AGENTS.md`; add a sibling `CLAUDE.md` containing `@AGENTS.md` so
  Claude Code loads the same rules.
- Use parallel agents only for independent tasks. Do not let multiple agents edit the same files concurrently; use
  separate branches or worktrees when parallel edits are necessary.

## Project invariants

PolyPhys is a solo-maintained, MIT-licensed Python package for polymer-physics and molecular-dynamics simulation-data
management and analysis, focused on bacterial chromosome organization (`amirhs1/poly-phys`).

- Preserve the artifact lineage: `segment → whole → ensemble_long → ensemble → space`.
- Preserve the phase/stage vocabulary:
  `simsAll, simsCont, logs, trjs, probes, analysis, viz, galaxy, allInOne` ×
  `segment, wholeSim, ens, ensAvg, space, galaxy`.
- Never rename, flatten, or simplify these conventions unless Amir explicitly asks. Filename-parsing changes require
  matching parser tests in the same change.
- Semantic versioning is used. The version lives only in `polyphys/__version__.py`; `pyproject.toml` reads it dynamically
  through `attr`. Change it only for a requested release.
- There is no PyPI release workflow. Distribution is GitHub source plus a Zenodo DOI. Do not add PyPI publishing,
  trusted publishing, or automated version bumps unless requested.

## Local Python environment and dependencies

The canonical local development environment is the named Conda environment `polylab_air`.

- Refer to it only by name. Never place an absolute Conda prefix or interpreter path in tracked files.
- Run every Python-dependent project command through:

  ```text
  conda run --no-capture-output -n polylab_air -- <command>
  ```

- Prefer module invocations, for example `conda run --no-capture-output -n polylab_air -- python -m pytest`.
- Do not use bare `python`, `pip`, `pytest`, `flake8`, `mypy`, `build`, or `sphinx-build`. Never use bare `pip`.
- Never install into Conda `base`, a system interpreter, another environment, or a user/global site. Never use `sudo`
  or `pip install --user`.
- Before any environment-changing command, show the exact command and obtain explicit approval. This includes package
  installation, updates, removals, and environment creation, recreation, cloning, export, or deletion.
- Install only dependencies declared in `pyproject.toml` unless Amir explicitly approves the dependency and its group.
- If `conda` or `polylab_air` is unavailable, stop and report it; do not silently fall back.
- Non-Python commands such as `git`, `gh`, `ls`, and `rg` do not require Conda.

## Python, documentation, and test standards

- Preserve the Python range declared in `pyproject.toml`; do not introduce incompatible syntax or APIs.
- Use NumPy-style docstrings with Sphinx/reStructuredText markup. The repository currently renders NumPy-style sections
  through Sphinx Napoleon; do not convert to Google style or add/switch documentation extensions unless requested.
- Use conventional sections as applicable: `Parameters`, `Returns` or `Yields`, `Raises`, `Warns`, `See Also`, `Notes`,
  `References`, and `Examples`.
- Keep type annotations authoritative. Docstrings explain semantics, units, array shapes, accepted ranges, side effects,
  and scientific assumptions rather than merely repeating types.
- Preserve Sphinx/reST roles and directives such as `:func:`, `:class:`, `:meth:`, `:mod:`, `:ref:`, `:cite:`, and
  `:math:`. Do not replace them with Markdown syntax inside docstrings.
- Every public function, method, and class must include a meaningful `Examples` section. Use small, deterministic,
  copyable examples; use `>>>` prompts when the example should execute as a doctest. Avoid network access, large data,
  and machine-specific paths.
- Every new or materially changed user-facing feature must also have a practical example in `docs/source/` or the
  appropriate README, not only in an API docstring.
- Every test function or test method must have a concise docstring stating the behavior, invariant, or regression it
  protects; do not merely restate the test name.
- Add or update tests for every behavior change and bug fix. Prefer focused regression tests and preserve numerical
  fixtures unless a documented scientific correction requires changing them.
- Docstring examples containing doctest prompts are executable tests. Update and run relevant doctests whenever those
  examples or their underlying behavior change.

## Validation and review

Run relevant existing, non-destructive checks without separate permission unless Amir explicitly asks to run them
himself. Start with the narrowest useful check, then broaden when warranted. Installation remains approval-required.

```bash
# Environment-changing; obtain explicit approval first.
conda run --no-capture-output -n polylab_air -- python -m pip install -e '.[dev]'

conda run --no-capture-output -n polylab_air -- python -m flake8 polyphys
conda run --no-capture-output -n polylab_air -- python -m mypy polyphys/analyze polyphys/manage
conda run --no-capture-output -n polylab_air -- \
  python -m pytest polyphys --cov=polyphys --cov-report=term-missing \
  --cov-report=xml --doctest-modules --doctest-glob='README.md'
conda run --no-capture-output -n polylab_air -- python -m build
```

- Pytest configuration already enables module and README doctests and sets `testpaths = ["polyphys/tests"]`. For a
  narrow first pass, target the changed test module or run `python -m pytest` through the required Conda prefix.
- For `docs/source/` changes, run relevant doctests and, when docs dependencies are installed:

  ```bash
  conda run --no-capture-output -n polylab_air -- python -m sphinx -b html docs/source docs/_build
  ```

  If dependencies are missing, propose the exact install command and obtain approval first.
- CI is the authoritative Python 3.11–3.13 matrix. Local checks may be narrower, but report their scope accurately.
- Review the final diff for correctness, unintended changes, stale docs, secrets, and generated artifacts.

## Scientific and performance correctness

- Preserve physical units, array-shape contracts, and numerical meaning. Explain changed expected values and their
  scientific basis in the PR.
- For correlated molecular-dynamics frames, use block averaging or another autocorrelation-aware uncertainty estimate,
  not naive per-frame standard errors.
- Prefer vectorized NumPy, pandas, and MDAnalysis operations over explicit Python loops when this improves performance
  without obscuring correctness.
- State time and space complexity in the docstring or developer docs for every new or materially changed core routine,
  especially code scaling with frames or particles.
- Cite a verifiable paper, textbook, standard, or official library documentation for physical models, statistical
  methods, algorithms, and nontrivial formulas. Put the citation where it is most useful: docstring, docs, or PR body.

## Branching, commits, and delivery

- The only long-lived branch is `main`; there is no `develop` branch. Never commit or push directly to `main`.
- Branch prefix determines the single required PR label:
  - `feature/*` → `enhancement`
  - `fix/*` or `hotfix/*` → `bug`
  - `refactor/*` or `chore/*` → `chore`
  - `deps/*` → `dependencies`
  - `docs/*` → `documentation`
  - `help/*` → `help wanted`
  - `question/*` → `question`
  - `release/*` → `release`
- A task-scoped branch may be created from `main` without asking. Never infer approval to stage, commit, push, open a
  PR, or merge from the original request or an earlier checkpoint.
- Before staging: show `git status --short` and `git diff`, summarize the proposed scope, and obtain explicit approval.
  Stage only approved files or hunks.
- Before committing: show `git diff --cached` and the proposed commit message, then obtain explicit approval.
- Before pushing or opening a PR: show the branch/base and proposed PR title/body, then obtain explicit approval. Never
  merge; Amir owns merges.
- Do not prefix commit or PR titles with an agent or tool name.
- When an agent materially authors or co-authors a commit, append exactly one matching trailer after a blank line:

  Codex:

  ```text
  Co-authored-by: Codex <noreply@openai.com>
  ```

  Claude Code:

  ```text
  Co-authored-by: Claude <noreply@anthropic.com>
  ```

- Verify the trailer with `git log -1 --pretty=%B` before pushing and after any amend.
- If branch prefix and intended PR label disagree, stop and ask. Never create or apply labels outside the approved list.

## Packaging

- Build backend: setuptools in `pyproject.toml`.
- Discovery includes `polyphys*` and excludes `notebooks*`, `projects*`, and `under_review*`.
- Optional groups are `docs`, `dev`, and `notebooks`. Install only what the task requires and only after approval.
- `polyphys.test_data` ships as package data. Put packaged fixtures there or deliberately update the package-data config.
- CLI entry point: `polyphys = "polyphys.cli:main"`.
- After changing packaging, package data, entry points, or version loading, run `python -m build` through Conda.

## Security and audit work

- Never expose, log, commit, or paste credentials, tokens, private keys, or sensitive datasets.
- Follow `SECURITY.md`: report suspected vulnerabilities privately; do not open a public issue or PR with exploit details.
- Audits are read-only by default. Do not modify, delete, clean, publish, or release unless explicitly requested.
- Install audit-only tools such as `pip-audit`, `bandit`, or `pip-licenses` only with explicit approval and only in a
  disposable `.audit-venv`, never in `polylab_air` or `pyproject.toml`.
- Retry a failed audit command once only for an obvious correction; otherwise record the error and continue with what
  can be verified.
- Separate verified, likely-but-untested, and unassessed findings. Never invent scores, coverage, performance results,
  or vulnerability status.
- Before creating or recommending a tracked issue, inspect open and closed issues when network access is available. If
  unavailable, state that duplication was not checked.

## Completion

- Update `AGENTS.md` for shared instructions. Update `CLAUDE.md` only when its import or a truly Claude-specific rule
  changes; never restore duplicated shared guidance.
- If a task changes a documented repository fact, update the relevant source documentation in the same change.
- In the final response, summarize changed files, why each changed, checks and results, skipped checks and reasons, and
  remaining risks or follow-up work.
- When implementation is complete, present the diff and stop before staging unless staging already has explicit approval.
