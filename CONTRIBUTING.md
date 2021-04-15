# Contributing to GTSFM

Contributions to GTSFM are welcome!  If you find a bug, please [open an issue](https://github.com/borglab/gtsfm/issues) describing the problem with steps to reproduce.  Better yet, if you figure out a fix, please open a pull request!

To open a pull request, here are some steps to get you started:

- [Fork the GTSFM repository](https://help.github.com/en/articles/fork-a-repo) and [clone to your machine](https://help.github.com/en/articles/cloning-a-repository).

- Create a branch for your changes.
  - `$ git checkout -b <name of your branch>`

- Validate that your changes do not break any existing unit tests. CI (Github Actions) should also pass.
  - Run all unit tests: `$ pytest tests`
 
- Reformat your code using Python [black](https://github.com/psf/black), with `-l 120` for a max line length of 120. 
- Ensure static analysis with flake8 does not throw any errors:

- Please provide documentation for any new code your pull request provides.


- [Open a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork) https://github.com/borglab/gtsfm/pulls from your branch.
  - Hint: having unit tests that validate your changes with your pull
    request will help to land your changes faster.

## Type Hints and Docstrings
Function signatures should include type hints and each function should be accompanied by a docstring.

## Coordinate System Conventions

Code in GTSFM adheres to a strict set of conventions about how rigid body transformations are expressed in code. A few examples are provided below:
- wTc:
- wTc_list:
- w_wUc:

We aks that contributors prefer GTSMA types wherever possible unless it's not already wrapped and is a lot of work to do so, or there are good advantages to using other types (like np arrays).

