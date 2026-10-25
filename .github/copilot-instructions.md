# Contribution and Coding Guidelines (Summary)

GTSFM's reconstruction pipeline is Python-based and depends heavily on GTSAM. Studio also includes a TypeScript/JavaScript frontend. Python depends on the stable `gtsam` release (`gtsam==4.3.0`); switch back to `gtsam-develop` only when GTSFM needs an API that is not in that release yet.

- Prefer GTSAM types unless there is a strong reason to use other types.
- Validate changes with the relevant Python tests; for Studio frontend changes, also run `npm run build --prefix visualization/frontend` and `npm run test:viewer --prefix visualization/frontend`.
- Format Python with `black -l 120 gtsfm tests` and sort imports with `isort --profile black -l 120 gtsfm tests`.
- Before committing or pushing Python changes, run the exact CI lint check from `.github/workflows/ci.yml`: `uv run flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503,E731 gtsfm tests`. Fix every reported violation and rerun until it exits successfully. Passing unit tests does not replace this check. An existing environment's `python -m flake8` may replace `uv run flake8` with the same arguments when preserving a working environment.
- Provide documentation for new code and include unit tests when appropriate.
- Follow naming conventions: `i` for camera indices, `j` for 3D point indices, `k` for measurement indices; use `wTi` for camera pose in world frame.
- Python style: type-hinted function signatures, early returns to avoid deep nesting, f-strings by default (except logging), no mutable default args, and avoid using dicts as objects (use NamedTuple instead).
- Docstrings required for each function, using Google-style format with a one-line summary, a blank line, and Args/Returns sections.
