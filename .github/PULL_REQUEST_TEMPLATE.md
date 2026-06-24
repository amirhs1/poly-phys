## Summary

-

## Checks

- [ ] `flake8 polyphys`
- [ ] `mypy polyphys`
- [ ] `pytest polyphys/tests polyphys --cov=polyphys --cov-report=term-missing --doctest-modules --doctest-glob="README.md"`
- [ ] `python -m build` if packaging metadata, package data, or package layout changed
- [ ] Docs build if `docs/source/` changed

## Scientific Correctness

- [ ] No measurement/statistics behavior changed
- [ ] Units, numerical fixtures, and domain assumptions are preserved or explained
- [ ] Parser lineage and organizer vocabulary are unchanged, or matching parser tests/docs were updated

## Notes

-
