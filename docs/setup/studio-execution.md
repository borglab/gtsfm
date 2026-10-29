# Studio execution configurations

Start Studio on your Mac with `gtsfm run`. Saved execution configurations stay
on that Mac; the browser remains connected to the Mac throughout a remote run.

## Configure once

The destination needs a GTSFM checkout and a compatible Python environment,
including Studio dependencies and the selected models. Configure passwordless
SSH access and verify the destination's host key once. Studio uses OpenSSH's
existing host aliases, identities, agent, jump hosts, and known-hosts database.
It does not prompt for passwords, install environments, or change host keys.

In **New run → Compute**, choose **Remote VM → SSH**. Enter the host or SSH alias,
remote checkout, and remote Python interpreter. A relative interpreter such as
`.venv/bin/python` resolves inside the remote checkout. For a Windows OpenSSH
destination running Linux computation, enter its WSL distribution as well.
Both absolute Linux paths and `~` paths work through WSL.

Select the device (`cpu` or, for example, `cuda:0`) and adjust **Machine settings**.
Choose **Check SSH connection** to inspect the destination; this temporary
connection stops after the check. Enter a name such as **Cyber** and choose
**Save configuration**. A local configuration can similarly be named **Mac**.

Configurations are stored in
`$XDG_CONFIG_HOME/gtsfm/execution-configurations.json`, or
`~/.config/gtsfm/execution-configurations.json` by default. Writes are atomic and
locked across Studio processes. Identity file paths are saved; private-key
contents and API/account tokens are not configuration fields.

## Run normally

1. Select a saved execution configuration. Its values populate the controls.
2. Upload a dataset or select a sample. Adjust the pipeline and machine settings.
3. Choose **Run reconstruction**, monitor **Activity / Logs**, and open **View results**.

Current controls are the submitted values. Editing them does not change the
saved copy until **Save configuration** is clicked. **Reload saved** restores
the saved values. A missing device stays visibly unavailable rather than
silently displaying a different device. Endpoint credentials must be re-entered
when switching API destinations; they are not saved with configurations.

Studio starts an authenticated, loopback-only remote API and SSH tunnel for the
run, uploads local input folders, and reuses the existing job/progress/cancel
protocol. It retrieves reconstruction exports, metrics, and splats to the Mac.
The viewer uses the original local photographs; remote-downloaded samples can
include their photographs in the result transfer. Results remain available
after the service and tunnel stop. **Stop**, or shutting down Studio, cancels
owned work and reaps its transports.

Readiness is attempted up to three times before submission. Startup errors are
kept in the job log; an ambiguous job submission is never automatically replayed.

## SSH worker clusters

To compute across existing SSH-accessible nodes, enter **SSH worker hosts**
(space-separated), **Scheduler host**, **Worker Python**, and **Worker device**.
When Studio targets a remote machine, that machine is the control node and
initiates these worker connections. The control node may be CPU-only.

Studio builds a native `distributed.SSHCluster` constructor from these controls.
Workers/threads/memory apply per worker host; **Scheduler port = 0** selects an
available port. The selected CPU/GPU binding is passed to each Dask Nanny's
worker environment. Each node needs the same software and shared input paths;
worker login configuration belongs on the control node. This starts ordinary
SSH workers, not a Slurm/PBS allocation or a cloud VM.

The PostgreSQL demo in `scripts/local_scheduler_db_remote_worker_demo.py` and
`gtsfm/utils/ssh_tunneling.py` was inspected. Its database initialization,
environment assumptions, and port-killing logic are not needed by Studio's
existing HTTP job path. Dask owns the actual scheduler/worker implementation;
Hydra converts constructor arguments to native Python containers.

The optional expert **Cluster config (YAML)** field remains available for existing
CLI/Studio configurations. It is not required to create or use saved UI settings.
See [native Dask configuration](dask-configuration.md) for that separate interface.

## Verification

`scripts/verify_studio_ssh.py` runs real HTTP-to-SSH reconstructions using a named
configuration, verifies retrieval of complete exports, checks exact repeat/reference
hashes, and cancels an unfinished run on failure. Run it against a disposable
Studio results workspace:

```bash
python scripts/verify_studio_ssh.py --configuration Cyber \
  --dataset /path/to/local/dataset --output /tmp/studio-ssh-verification.json
```

The verification CLI defaults to the exact `sift_brute_force` correspondence
preset. Use the same effective matcher in both arms; approximate FLANN matching
can vary between repeats. Frontend presets apply to cached and uncached models.

This is a functional/equivalence check. A scientific timing comparison additionally
requires pinned inputs, source, models, seeds, cache policy, hardware, resource
limits, repeated baseline/treatment trials, and telemetry. It must not be inferred
from a successful launch or a single runtime measurement.

See the [verification report](../benchmarks/studio_ssh_verification.md) for observed
acceptance results, exact export comparisons, retained failures, and limits.
