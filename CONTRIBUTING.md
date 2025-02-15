# Contributing to GTSFM

Contributions to GTSFM are welcome!  If you find a bug, please [open an issue](https://github.com/borglab/gtsfm/issues) describing the problem with steps to reproduce.  Better yet, if you figure out a fix, please open a pull request!

To open a pull request, here are some steps to get you started:

- [Fork the GTSFM repository](https://help.github.com/en/articles/fork-a-repo) and [clone to your machine](https://help.github.com/en/articles/cloning-a-repository).

- Create a branch for your changes.
  - `$ git checkout -b <name of your branch>`

- Validate that your changes do not break any existing unit tests. CI (Github Actions) should also pass. We use `pytest`.
  - Run all unit tests: `$ pytest tests`
 
- Reformat your code using Python [black](https://github.com/psf/black), with `-l 120` for a max line length of 120: `$ black -l 120 gtsfm tests` 
- Auto-sort your imports using [isort](https://pycqa.github.io/isort/), with `--profile black` to avoid conflicts with `black` and `-l 120` for a max line length of 120: `$ isort --profile black -l 120 gtsfm tests`
- Ensure static analysis with flake8 does not throw any errors: `$ flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503,E731 gtsfm tests`

- Please provide documentation for any new code your pull request provides.

- [Open a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork) https://github.com/borglab/gtsfm/pulls from your branch.
  - Hint: having unit tests that validate your changes with your pull
    request will help to land your changes faster.

## SfM Variable Name Conventions
We use a specific naming convention for image/SfM data throughout the codebase:
- `i` for camera indices
- `j` for 3d point indices
- `k` for measurement indices

## Coordinate System Conventions

Code in GTSFM adheres to a strict set of conventions about how rigid body transformations are expressed in code (described [here](https://gtsam.org/gtsam.org/2020/06/28/gtsam-conventions.html)). A few examples are provided below:
- `wTi`: pose of the i'th camera in the world frame. If there is only a single camera, `wTc` is also acceptable.
- `wTi_list`: pose of the i'th camera in the world frame, for n cameras
- `w_i2Ui1`: a ray from camera i2 to camera i1, inside the world coordinate system. Mathematically, `w_i2Ui1 = wRi_list[i2].rotate(i2Ui1)`. Since `i2Ui1` is the unit-normalized translational component of the pose of camera i1 inside i2's frame, then intuitively this is the ray from i2 to i1.

We ask that contributors prefer [GTSAM types](https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/geometry.i) wherever possible unless it's not already wrapped and is a lot of work to do so, or there are good advantages to using other types (like np arrays).

## Python Style
- **Auto-Formatting**: We format code with `black` and a maximum line length of 120 characters.
- **Type hints**: Function signatures should include type hints. Do not put type information in the docstring if it is redundant with the type hint.
- **Branch Logic**: Return early, don't nest.
- **f-strings**: Use f-strings as a default for regular strings, except when logging, where there is a specific logging format.
- Default arguments should never be mutable (these will lead to unexpected and strange behavior), `def foo(mylist = [])` is not ok
- Do not use a dictionary as an object -- use `NamedTuple` instead.
- **Imports**: We use [isort](https://pycqa.github.io/isort/) to handle formatting imports automatically. See the [Google python style guide](https://google.github.io/styleguide/pyguide.html#313-imports-formatting) for more details.
- **Docstrings Required**: Each function should be accompanied by a docstring. Docstrings should be added as [described here](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
    - Docstrings should start with a one-line summary of the program terminated by a period.
    - Leave one blank line.
    - If the input or return arguments are not None, use the following syntax:
```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table, keys: Sequence[Union[bytes, str]], require_all_keys: bool = False
) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: Optional; If require_all_keys is True only
          rows with values set for all keys will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

We provide a `.vscode/settings.json` and pyproject.toml which configure your vscode environment to use our formatting 
settings. This would also need installing some vscode extensions to work as expected: Flake8, Code Spell Checker, Black 
formatter, and Pylint. 
