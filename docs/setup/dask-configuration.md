# Dask configuration

For locally saved settings edited in Studio, including automatic remote startup,
use [Studio execution configurations](studio-execution.md). The constructor-file
interface below is an optional expert/CLI path.

Use the existing **Cluster config (YAML)** field in Studio, or the runner's
`--cluster_config` argument, to reuse a local cluster configuration. GTSfM loads
the file with its existing Hydra/OmegaConf dependency and instantiates Dask's
`LocalCluster`. There is no separate compute-profile format or profile database.

```yaml
# local-cluster.yaml: Hydra constructor configuration
_target_: distributed.LocalCluster
n_workers: 1
threads_per_worker: 2
processes: true
memory_limit: 18GiB
local_directory: /tmp/gtsfm-dask
dashboard_address: ":8787"
```

Enter the file's path on the machine running GTSfM. `~` and existing packaged
config names are supported; `cluster_local.yaml` is a minimal packaged example.
The explicit constructor arguments in this file take precedence over Studio's
manual worker/thread/memory fields. Dask parses memory limits, including `auto`,
units and zero (unlimited). Omitted arguments use Dask's defaults and runtime
configuration. Set `local_directory` explicitly when the file must determine
scratch storage: the legacy `--dask_tmpdir` option still sets Dask's global
temporary-directory policy.

Use **native Dask configuration** for runtime policies. This is a separate file
from the Hydra constructor configuration above:

```yaml
# /path/to/dask-config/distributed.yaml
distributed:
  worker:
    memory:
      target: 0.60
      spill: 0.70
      pause: 0.80
      terminate: 0.95
```

Start GTSfM with `DASK_CONFIG=/path/to/dask-config` in its environment. Dask also
supports environment keys such as
`DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=127s`. Set these before launching
Studio/the runner so the scheduler and worker processes inherit them. GTSfM
registers its existing 300-second communication and disabled-worker-TTL
fallbacks through Dask's APIs, then reapplies Dask's collected YAML/environment
values. For programmatic callers, apply `dask.config.set(...)` after importing
`gtsfm.runner`; import-time collection does not track the provenance of earlier
programmatic overrides.

Leaving **Cluster config** empty retains the existing manual controls. Legacy
`workers:` YAML continues to select the existing SSH/GPU routes and uses their
existing manual arguments. Modal retains its existing configuration flow.
Real remote SSH/dask-cuda deployments are not covered by the local verification.

The experiment used the repository's Dask/distributed **2025.9.1**, without a
dependency upgrade. Consult the installed version when newer stable docs differ.

- [Dask configuration, YAML and environment variables](https://docs.dask.org/en/stable/configuration.html)
- [Dask LocalCluster constructor](https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster)
- [Hydra object instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
- [Verification protocol and results](../benchmarks/native_dask_verification.md)
