# GitHub Actions workflows

| Workflow | Triggers | Purpose and outputs |
| --- | --- | --- |
| [ci.yml](ci.yml) | Pull requests, except changes only under `gtsfm/visualization/**`; manual dispatch | Runs Flake8 with a 120-column limit and unit tests with coverage. A separate job runs reconstruction benchmarks and uploads metrics, plots, results, and Dask reports. PR titles containing `[skip benchmarks]` skip the benchmark job. |
| [test-reproducibility.yml](test-reproducibility.yml) | Pull requests; manual dispatch | Runs `tests/repro_tests` to check algorithm reproducibility. |
| [benchmark-self-hosted.yml](benchmark-self-hosted.yml) | Manual dispatch | Runs the configured dataset benchmarks on a self-hosted runner and uploads metrics, plots, and reconstruction results. It expects the existing `gtsfm-v1` environment and datasets under `/usr/local/gtsfm-data`. Run the shorter CI benchmarks first. |
| [build-wheels.yml](build-wheels.yml) | Manual dispatch | Builds wheels and source distributions on macOS, Linux, and Windows with Python 3.12, and uploads them as workflow artifacts. It does not publish them to PyPI. |
| [pages.yml](pages.yml) | Pushes to `master` changing `paper/**` or `.github/workflows/pages.yml`; manual dispatch | Uploads the `paper` directory and deploys it to GitHub Pages. |

To run a workflow manually, open the repository's **Actions** tab, select the
workflow, and choose **Run workflow**. Check the self-hosted runner's availability
and dataset setup before scheduling its longer benchmarks.

The CI and reproducibility workflows install the system Graphviz package.
Python's `pydot` package uses the Graphviz `dot` executable to render process
graphs, so `dot` must also be installed when running those tests locally.
