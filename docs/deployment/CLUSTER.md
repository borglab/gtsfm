# GTSFM on CLUSTER

## How to Run GTSfM on a Cluster?

GTSfM uses the [SSHCluster](https://docs.dask.org/en/stable/deploying-ssh.html#dask.distributed.SSHCluster) module of [Dask](https://distributed.dask.org/en/stable/) to provide cluster-utilization functionality for SfM execution. This readme is a step-by-step guide on how to set up your machines for a successful run on a cluster.

1. Choose which machine will serve as the scheduler. The data only needs to be on the scheduler node.
2. Create a config file listing the IP addresses of cluster machines (example in [gtsfm/configs/cluster.yaml](https://github.com/borglab/gtsfm/blob/master/gtsfm/configs/cluster.yaml)).
    - Note that the first worker in the cluster.yaml file must be the scheduler machine where the data is hosted.
3. Enable passwordless SSH between all the workers (machines) on the cluster.
    - Log in individually to each machine listed in the cluster config file.
    - For each of the other machines on the cluster, run:
        * ```bash
          ssh-copy-id {username}@{machine_ip_address_of_another_worker}
          ```
        * If you see `/usr/bin/ssh-copy-id: ERROR: No identities found`, then run `ssh-keygen -t rsa` first.
        * Repeat the two steps above on all machines.
    - Note machines should be able to ssh into themselves passwordless e.g. host1 should be able to ssh into host1.
    - If the cluster has 5 machines, then `ssh-copy-id` must be run 5*5=25 times.
4. Install the same GTSFM revision and dependencies on every node, with the checkout at an identical absolute path. Follow either the [Conda setup guide](../setup/conda-setup.md) or the **development environment** path in the [uv setup guide](../setup/uv-setup.md), including submodule initialization and an editable installation. Use the same installation method across nodes.
5. Make that environment's Python available to non-interactive SSH sessions on every node. Dask starts workers through these sessions.
    - For Conda, initialize Conda in the shell startup file and add `conda activate gtsfm-v1` after its initialization block.
    - For a uv development environment, add `export PATH="/home/username/gtsfm/.venv/bin:$PATH"`, using the actual absolute checkout path.
    - Place this environment setup before any early return for non-interactive shells, such as:
      ```bash
      case $- in
          *i*) ;;
          *) return;;
      esac
      ```
    - From another node, verify the selected interpreter and package import:
      ```bash
      ssh username@host 'python -c "import sys, gtsfm; print(sys.executable); print(gtsfm.__file__)"'
      ```
      The interpreter must belong to the intended Conda environment or `.venv`. An editable installation makes GTSFM importable without a separate `PYTHONPATH` override.

6. Add the host keys to your `known_hosts` file.
    - On  cluster machine, run the following commands in  terminal.
    This will SSH to each machine, and you will be prompted to trust its key
    ```
    ssh username@localhost
    ```
    You will likely see a message like: `The authenticity of host 'localhost (::1)' can't be established. ... Are you sure you want to continue connecting (yes/no/[fingerprint])?`
    Type `yes` and press Enter. Then you can exit

    - After doing so, the keys will be saved in ~/.ssh/known_hosts
    - Otherwise, you would need set up SSHCluster with extra options `connect_options={"known_hosts": None}`



7. Update the weight
    - ```bash
        scripts/download_model_weights.sh
         ```

8. Log into scheduler machine again and download the data to scheduler machine.

9. edit the config file `cluster.yaml`

9. Run GTSFM using the selected environment (activate Conda, or use `uv run gtsfm` for a uv development environment) with `--cluster_config`, for example
    - ```
      gtsfm execute -- --loader colmap --dataset_dir /home/username/gtsfm/skydio-32 --config_name sift_front_end.yaml --cluster_config cluster.yaml
      ```
    - Always provide absolute paths for all directories

10. If you would like to check out the dask dashboard, you will need to do port forwarding from machine to your local computer
    - ```
      ssh -N -f -L localhost:local_port:localhost:machine_port username@machine_adress
      ```

11. The results will be generated on the scheduler machine. If you would like to download results from the scheduler machine to your local computer
    - ```
      scp -r username@host:machine/results/path /local/computer/directory
      ```
Or you can take advantage of the vs-code port, in the scheduler machine to check report

> **Note:** Please utilize `gtsfm/utils/ssh_passwordless_setup.py` to facilitate the set up

## How to Run GTSfM on machine with multiple GPUs
### Use PACE Phoenix, Slurm as example


- Use `salloc` to request resource, then you don't need `dask-jobqueue`, which is for submitting jobs from within code. 
- Slurm would automatically set up `CUDA_VISIBLE_DEVICES`
- Memory settings should be slightly less than allocation per GPU to leave headroom

- check results with `gtsfm viz` (or `uv run gtsfm viz`) on log in node, not on computation node

### Cluster Workflow, on `gtsfm/runner.py`

```bash
run() → _create_dask_client()
│
├─ if cluster_config?
│  │
│  ├─ NO ──────────────────────────> [1] LocalCluster (CPU/GPU)
│  │                                   
│  │
│  └─ YES → setup_ssh_cluster_with_retries()
│           │
│           ├─ check: is_gpu_cluster?
│           │
│           ├─ NO ───────────────────> [2] SSHCluster 
│           │                           Multi Machines with GPU on each
│           │                           Use cluster.yaml
│           │                           workers: [host1, host2, host3]
│           │
│           └─ YES → _create_gpu_cluster()
│                    │
│                    ├─ check: unique_hosts == 1?
│                    │
│                    ├─ YES ──────────> [3] LocalCUDACluster
│                    │                   Single Machine and Multi GPU (PACE)
│                    │                   Use cluster_gpu.yaml
│                    │                   workers: [localhost, localhost]
│                    │
│                    └─ NO ───────────> [4] NotImplementedError
│                                        Multi Machines and Multi GPUs on each (Not Support)
```