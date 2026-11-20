# GTSfM Studio experiments

Start Studio on the Mac with `gtsfm run`. Named execution configurations stay on
that Mac and populate the editable controls used for submission.

## One experiment, from inputs to evidence

In **New experiment**, choose an image folder (or a sample/previous upload), a
reconstruction preset, and a compute destination. The standard presets and machine
defaults are visible together; the full preset catalog, Hydra overrides and worker
settings are under the corresponding advanced controls. A folder of images can be
used directly with the Olsson loader; a dataset containing `images/` also works.
Studio displays actual capture previews and reads image headers automatically.

**Run experiment** performs preparation before submitting the reconstruction.
**Experiments** keeps each run's progress, geometry, camera photographs, scalar
metrics, resolved pipeline configuration, logs and downloads in the same context.
New runs record the source content fingerprint, Python/package versions and SHA-256
input manifest. A checkout also records its Git commit and whether it has local
changes. An installed wheel fingerprints its installed files. Older runs
explicitly report missing provenance.

**Use these settings** makes an editable copy without starting a job. The experiment
record downloads as `gtsfm.experiment.v1` JSON, including measured metrics and the
resolved configuration when available. **Import experiment** applies its pipeline
settings and removes account credentials, machine destinations and absolute input
paths. Select local images or the named sample and review compute before Run.
The record identifies source code; it does not bundle a dirty checkout or model
weights. A copied run uses the installed code and does not promise bitwise identity.

Comparison shows recorded metrics side by side. A matching input fingerprint is
required before Studio calls the datasets matched; users must still check presets,
code and other conditions before treating differences as benchmark evidence.
No metrics are synthesized when a pipeline did not save them.

The visible **Run time limit (minutes)** defaults to 60 and requests cancellation
of owned jobs at the deadline, including queued time. Preparation of the compute
workspace occurs before this timer starts. Remote cancellation is a request and
must reach the provider; it is not a guaranteed dollar spending cap. Normal Studio
shutdown also cancels owned jobs. Recovered history is never automatically replayed.

## Modal: browser sign-in and automatic preparation

Choose **Modal cloud**, then **Connect saved Modal account** to verify an existing
native profile, or **Sign in to Modal** for first-time setup. Studio invokes Modal's
supported [`modal token new --no-activate`](https://modal.com/docs/cli/latest/token)
browser authorization flow. Follow **Continue to Modal** and finish sign-in/approval
there. Profile credentials stay in Modal's native configuration on the Studio host;
the browser sends a profile name rather than asking users to paste API keys.
Reading another profile and a new login do not change the native active-profile
selection. A fresh browser login can replace credentials for that named workspace.
Login expires after five minutes and its child process is stopped on Studio shutdown.

This is Modal's browser login with native local credentials. Organization-wide
[SAML SSO](https://modal.com/docs/guide/saml-sso) is a separate Modal Enterprise
feature configured by the workspace administrator. Studio does not implement a
separate identity provider or claim a custom OAuth integration.

Choose an available GPU and **Run experiment**. Studio builds the current checkout,
checks the CPU control service, then stages inputs and submits computation. Matching
completed deployments are reused within the running Studio host when source,
account and resources match; health checks still run. Matching uploaded dataset
contents can be reused on the remote host before another transfer. Account
connection alone does not start a GPU reconstruction.

Native Modal Images, Functions, Volume, Dict and Secret objects provide execution.
HTTP/control traffic uses CPU; only reconstruction uses the single-GPU function.
The Modal device is normalized to `cuda:0`, including legacy saved CPU configurations.
The provider also rejects unsupported devices before spawning a call. Finished jobs
transfer authenticated result archives back to the Studio host. Input paths remain
local in the submitted record; executed source/runtime identity comes from the worker
when recorded. Source-photo resolution uses original uploaded inputs or downloaded
sample imagery.

Use Modal SDK 1.6.3 or newer for the selected image builder; `uv sync` installs the
locked integration. GPU access, account approval and available capacity are controlled
by Modal. A deployed and healthy control service does not prove GPU execution or
successful result transfer. Verify the FunctionCall and GPU-container state after
completion/cancellation; a configured scale-down window alone is not E2E evidence.

## Local persistence and native Dask workers

Settings use atomic, cross-process-locked writes in
`$XDG_CONFIG_HOME/gtsfm/execution-configurations.json`, or
`~/.config/gtsfm/execution-configurations.json`. Modal credentials stay in the
native Modal profile. Saved local and Modal configurations from earlier drafts
remain readable. Legacy Ray presets stay on disk and are hidden in Studio.

For additional SSH workers, enter **SSH worker hosts**, **Scheduler host**,
**Worker Python** and the worker device. Studio produces the native
`distributed.SSHCluster` constructor. Nodes need compatible software and shared
input paths; the control machine owns worker SSH authentication. This is not a
Slurm/PBS allocation. The optional expert YAML field remains available; normal
saved UI settings do not require authoring YAML or entering a configuration path.

See [native Dask configuration](dask-configuration.md).

## Verification

Start a real experiment in Studio, then verify its returned results with:

```bash
python scripts/verify_studio_run.py --studio-url http://127.0.0.1:5173 \
  --job-id YOUR_RUN_ID --expected-cameras 12 \
  --output /tmp/studio-acceptance/report.json
```

Add `--require-splats` for a run that includes Gaussian splat training. The command
does not submit computation. It waits for that run, checks nonempty geometry and
the expected camera count, decodes every thumbnail and preview, requires source
and input fingerprints plus resolved configuration and measured metrics, checks
that the result archive contains the displayed geometry and original photos, and
round-trips the experiment record without starting another job. For a protected
Studio host, provide its bearer key through `GTSFM_STUDIO_API_KEY`.

Browser rendering, first-time account approval, and native provider cancellation
and GPU scale-down require separate observation. A successful HTTP response or a
local cancellation acknowledgement is insufficient evidence for those cases.
