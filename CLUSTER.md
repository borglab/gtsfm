# GTSFM on CLUSTER

### How to Run GTSfM on a Cluster?

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
4. Clone gtsfm and follow the main readme file to setup the environment on all nodes in the cluster at an identical path
    - ```bash
        git clone --recursive https://github.com/borglab/gtsfm.git
        conda env create -f environment_linux.yml
        conda activate gtsfm-v1
         ```
5. Log into cluster machines to set up conda environment initialize 
    - edit `bashrc` using `nano ~/.bashrc ` or `code ~/.bashrc` by VS code
    - find the section `# >>> conda initialize >>> ` and `conda activate gtsfm-v1`
    - the section of `# <<< conda initialize <<<` to be placed in front of 
    ```
    # If not running interactively, don't do anything
    case $- in
        *i*) ;;
        *) return;;
    esac
    ```

 
    - After the above set up, when ssh into another machine, the environment would be automatically `gtsfm-v1`
    - `which python` and `echo $CONDA_PREFIX` should point to the same envs

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
        download_model_weights.sh
         ```

8. Log into scheduler machine again and download the data to scheduler machine.

9. edit the config file `cluster.yaml`

9. Run gtsfm with `--cluster_config` flag enabled, for example
    - ```
      ./run --loader colmap --dataset_dir /home/username/gtsfm/skydio-32 --config_name sift_front_end.yaml --cluster_config cluster.yaml
      ```
    - Always provide absolute paths for all directories

10. If you would like to check out the dask dashboard, you will need to do port forwarding from machine to your local computer:
    - ```
      ssh -N -f -L localhost:local_port:localhost:machine_port username@machine_adress
      ```

11. The results will be generated on the scheduler machine. If you would like to download results from the scheduler machine to your local computer:
    - ```
      scp -r username@host:machine/results/path /local/computer/directory
      ```
PS. Please utilize `gtsfm/utils/ssh_passwordless_setup.py` to facilitate the set up