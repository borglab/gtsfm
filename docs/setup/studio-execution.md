# Studio execution configurations

Start Studio on the Mac with `gtsfm run`. Named execution configurations stay on
that Mac and populate the editable controls used for submission.

## SSH machines: standard Ray deployment

Studio uses Ray 2.59.0's [Cluster Launcher](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html),
[Jobs service](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html),
dashboard forwarding and rsync commands. There is no Studio-specific remote API bootstrap or SSH tunnel
implementation. Dask still performs the reconstruction computation inside the
ordinary command supervised by Ray.

Install the optional local integration once:

```bash
uv sync --extra ssh
```

On each destination, install a compatible GTSFM checkout/environment and the
required model submodules once. Keep the Ray tool separate from that environment:

```bash
uv tool install --python 3.12 'ray[default]==2.59.0'
```

The destination must provide a **direct Linux SSH endpoint**, `rsync`, and a
working GPU driver/CUDA environment when GPU computation is selected. For WSL,
use its Linux Tailscale SSH address; the Windows `wsl.exe` shell adapter is no
longer supported. Complete any Tailscale SSH identity check normally. Studio does
not change SSH services, drivers or network policy.

In **New run → Compute → Remote VM → SSH**, enter:

- Host or existing OpenSSH host alias, plus the Linux username.
- Existing remote GTSFM checkout and its Python interpreter. Relative Python
  paths resolve within that checkout; remote `~/` paths are supported.
- Remote Ray executable, normally `~/.local/bin/ray` after the installation above.
- Device and worker/thread/memory settings under **Machine settings**.

Ray's launcher uses its own OpenSSH configuration and identity handling. A
non-default SSH port must be configured in an OpenSSH host alias; it is rejected
as a separate port setting. Ray 2.59's launcher uses its standard host-key options,
which differ from Studio's previous strict-known-hosts launcher. Use a trusted
private network such as an authenticated Tailscale endpoint.

**Check SSH connection** starts or reuses the dedicated Ray runtime and runs a
short inspection job. Ray assigns the physical GPU at launch; `cuda:0` denotes
the GPU visible inside that allocation. The UI does not promise a particular
physical GPU index. CPU selections remain explicit. Requests for more Ray CPUs
than the remote machine reports fail with a settings error instead of waiting
indefinitely for impossible resources.

Choose a name such as **Cyber**, then **Save configuration**. Local computation
can likewise be named **Mac**. Selecting a saved configuration fills the controls;
edits affect the next run and are saved only when **Save configuration** is clicked.
**Reload saved** restores the stored copy. Unavailable devices remain visible and
cannot be submitted silently.

## Run and manage machines

1. Select a saved configuration and upload a dataset or choose a sample.
2. Choose **Run reconstruction**. Ray stages inputs, supervises the pipeline and
   streams logs; Studio presents progress and retrieves complete results.
3. Open **View results** on the Mac. Geometry, metrics, splats and source-photo
   metadata remain usable after disconnecting.

**Stop**, or normal Studio shutdown, requests cancellation of owned Ray jobs and
closes their local dashboard transports. The reusable Ray runtime remains on the
machine for subsequent runs. Studio never automatically stops another runtime,
replays an uncertain submission, or installs packages into the project Python.

Canonical launcher configurations live beneath the results workspace at
`.gtsfm/ray/`. Use Ray's own commands to inspect or explicitly tear down a runtime
when no jobs should remain:

```bash
ray exec /path/to/cluster.yaml '~/.local/bin/ray status'
ray down -y /path/to/cluster.yaml
```

Only tear down a runtime you own and intend to stop. Uploaded datasets, results
and package/model caches remain on disk; they are not an automatic retention or
uninstall policy. Native Ray Jobs SDK requests have no public per-request timeout
option in this release, so a wedged dashboard can delay status/cancellation.
Normal completion, cancellation and disconnection must be verified separately.

## Modal: beginner account workflow

Use an ordinary Modal **Starter** workspace and finish the official CLI-token
approval after account sign-in. GPU access may require a payment method on Modal.
Enter credentials only in the local Studio credential controls; they are not
saved with named configurations.

Select **Remote VM → API → Modal**, choose an available GPU, and use **Set up &
deploy** on a new workspace. Studio packages the checkout with native Modal Images,
Functions, Volume, Dict and Secret objects. HTTP/control traffic uses a small CPU
function; reconstruction alone uses the GPU function. Readiness is not proof of
actual GPU execution or successful result transfer.

A fresh Modal selection initializes `cuda:0`. Loaded or explicitly chosen
unavailable devices stay visible and block submission. The control service also
rejects unsupported devices before spawning a GPU call. Completed jobs transfer
full authenticated result archives back to the Mac, including source photos for
remote samples. The local viewer does not need public photo endpoints.

Stop an active test from Studio and independently verify its Modal FunctionCall
and GPU-container state. Starter credits, account approval, GPU availability and
actual charges must be checked in the authenticated workspace. A deployment,
mock test or configured scale-down window alone does not establish E2E success.

## Local persistence and native Dask workers

Settings use atomic, cross-process-locked writes in
`$XDG_CONFIG_HOME/gtsfm/execution-configurations.json`, or
`~/.config/gtsfm/execution-configurations.json`. Identity paths are saved;
private-key contents and API/account credentials are excluded. Changing an API
destination requires its credentials again.

For additional SSH workers, enter **SSH worker hosts**, **Scheduler host**,
**Worker Python** and the worker device. Studio produces the native
`distributed.SSHCluster` constructor. Nodes need compatible software and shared
input paths; the control machine owns worker SSH authentication. This is not a
Slurm/PBS allocation. The optional expert YAML field remains available; normal
saved UI settings do not require authoring YAML or entering a configuration path.

The older PostgreSQL scheduler demo was inspected; its database setup and
port-killing assumptions are not used. See [native Dask configuration](dask-configuration.md).

## Verification

`scripts/verify_studio_ssh.py` exercises a saved SSH configuration through real
Studio HTTP requests and compares complete retrieved export hashes. It defaults
to exact `sift_brute_force` correspondence matching:

```bash
python scripts/verify_studio_ssh.py --configuration Cyber \
  --dataset /path/to/local/dataset --output /tmp/studio-ssh-verification.json
```

This is a functional/equivalence check. Performance or regression comparisons
require pinned source, inputs, assets, settings, seeds, cache policy, hardware,
resource limits and telemetry. Cross-GPU differences do not alone establish a
regression. [Earlier SSH evidence](../benchmarks/studio_ssh_verification.md) is
explicitly historical and does not validate the replacement Ray or Modal paths.
