# third_party_reference/ts2vec/

**This is NOT the installed dependency.** It is a read-only reference
copy of the core TS2Vec source files (`ts2vec.py`, `models/`,
`utils.py`), vendored here for two purposes only:

1. **Human/AI reading reference** — so `src/models/ts2vec_wrapper.py`
   (M7) can be written and reviewed against the exact API this project
   targets, without needing network access to browse GitHub.
2. **Emergency fallback** — if `pip install git+https://github.com/zhihanyue/ts2vec.git@b0088e14a99706c05451316dc6db8d3da9351163`
   ever fails (e.g. the commit becomes unreachable due to a force-push
   or repo deletion upstream), this copy documents exactly what code
   was pinned, so a fallback fork (see `configs/base.yaml` →
   `ts2vec.fallback_fork`) can be created from it.

**The actual runtime dependency is installed via `requirements.txt` /
`environment.yml`, pinned to the commit hash above — NOT imported from
this folder.** `src/models/ts2vec_wrapper.py` does `from ts2vec import
TS2Vec` (the pip-installed package), not a relative import from here.

## Provenance

- Source: `ts2vec-main.zip`, provided by the project author on
  2026-07-03 (downloaded from GitHub's "Download ZIP" button, not
  `git clone` — no `.git` metadata was present in the zip).
- Pinned commit hash (per author): `b0088e14a99706c05451316dc6db8d3da9351163`
- Verified: `ts2vec.py` content in this zip is textually identical to
  `https://github.com/zhihanyue/ts2vec/blob/main/ts2vec.py` as fetched
  on 2026-07-03 (current `main` HEAD at fetch time:
  `ac76ac278b564e81010cf07d14d4109e9d202ead` — different commit hash
  from the pinned one, but identical file content for the core module).
- Files excluded from this vendored copy (not needed by the wrapper,
  present in the original repo): `datasets/`, `scripts/`, `tasks/`,
  `train.py`, `datautils.py` — these are TS2Vec's own benchmark/eval
  harness for UCR/UEA/ETT datasets, unrelated to this project's usage
  (we only import the `TS2Vec` class and train/encode on our own data).

## License

TS2Vec is MIT-licensed (see `LICENSE` in this folder). This vendoring
is for reproducibility/reference purposes and complies with the
license's redistribution terms.
