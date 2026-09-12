# GTSFM Studio visualization

This folder contains the Studio web application, frontend, and remote deployment code.
See the [project README](../README.md) for installation and local development.

## Modal deployment

In Studio, choose **Remote VM** and select Modal. Enter your Modal token ID and
token secret separately, or paste the full `modal token set --token-id … --token-secret …`
command into either field. If your account does not already have a deployment,
click **Set up & deploy Modal workspace**.

[modal_deployment.py](modal_deployment.py) runs the Modal deployment command,
streams setup progress, and discovers and verifies the deployed endpoint.
[modal_app.py](modal_app.py) defines the protected control service, GPU job, and
persistent `gtsfm-studio-data` volume.

### Image contents and caching

The GPU image definition in `modal_app.py`:

1. Starts from NVIDIA's `nvidia/cuda:12.8.1-devel-ubuntu22.04` image and adds Python 3.12.
2. Installs system packages, including compilers, graphics libraries, and Graphviz.
3. Installs Python dependencies from [pyproject.toml](../pyproject.toml) and
   [uv.lock](../uv.lock) using `uv_sync` with a frozen lockfile and no default groups.
4. Adds the checkout's `gtsfm`, `visualization`, and `thirdparty` source directories
   after the dependency layers.

Modal caches unchanged image layers. The first build or a dependency change may
take longer; application source changes can reuse the unchanged dependency layers.
Model weights and runtime caches are stored under `/mnt/gtsfm-studio/cache` on the
persistent volume.

There is no separately published GTSFM runtime image, custom Dockerfile, or
GTSFM image registry to configure. The NVIDIA base image still comes from its
upstream registry. Change the image definition in `modal_app.py` when the deployment
needs different system dependencies.
