# Contribution and Coding Guidelines (Summary)

This is a Python-only but GTSAM-heavy project. Any C++ changes are made in GTSAM, and pulled in via gtsam-develop.

- Prefer GTSAM types unless there is a strong reason to use other types.
- Validate changes by running unit tests with `pytest tests`; CI should pass.
- Format Python with `black -l 120 gtsfm tests` and sort imports with `isort --profile black -l 120 gtsfm tests`.
- Run static analysis with `flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503,E731 gtsfm tests`.
- Provide documentation for new code and include unit tests when appropriate.
- Follow naming conventions: `i` for camera indices, `j` for 3D point indices, `k` for measurement indices; use `wTi` for camera pose in world frame.
- Python style: type-hinted function signatures, early returns to avoid deep nesting, f-strings by default (except logging), no mutable default args, and avoid using dicts as objects (use NamedTuple instead).
- Docstrings required for each function, using Google-style format with a one-line summary, a blank line, and Args/Returns sections.
