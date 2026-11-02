# Studio Ray and Modal acceptance, 2026-10-30

This follow-up tests the native Ray replacement introduced after `d17530ee`.
It does not reinterpret the earlier custom-transport runs as Ray evidence.

## Observed SSH execution

The Mac and the Linux RTX 3080 endpoint joined the user's personal Tailscale
network. The endpoint used Ubuntu's packaged OpenSSH server with public-key
authentication on its private Tailscale address. Ray 2.59.0 ran from a separate
`uv tool` environment; the existing GTSFM Python environment and GPU driver were
preserved. Local OpenSSH configuration matched both the DNS name and resolved IP.

The production Studio UI exercised saved settings, save/reload, native Ray
inspection, input staging, submission, streamed logs, and returned results:

- Run `113126d2ed30`: four-image Olsson fixture, VGGT with SIFT brute-force
  correspondences, resolution 518, no splats, one worker, two threads, and an
  18 GiB Dask worker memory setting.
- Actual logs confirmed VGGT inference on `cuda:0`. The completed retriangulated
  result had **four cameras and 1,509 tracks**.
- All **10 export hashes** matched the historical fixture's `F1` exports.
  This is fixture parity, not a general accuracy or performance claim.
- The Mac browser loaded all four source thumbnails, rendered the returned
  cameras, and displayed 216 lines of runner logs without browser errors.
- Native `ray down` stopped the test head. A subsequent Studio connection
  started it again and launched another reconstruction.
- Studio Stop cancelled runs `945241612755` and `815c60d48614`. The latter was
  stopped during reconstruction while the GPU held 4,102 MiB. Independent native
  Jobs SDK inspection reported both as `STOPPED`.
- After cancellation, native Ray reported zero CPU/GPU reservations and no
  pending demands. NVIDIA telemetry returned to 964 MiB and 0% utilization,
  the observed desktop idle level. No local dashboard CLI remained.
- After Studio shutdown, the Ray head remained available. Restarting Studio
  preserved the completed run's local geometry, logs, and four source photos;
  the browser acceptance check passed again.

The WSL endpoint initially stalled on larger SSH transfers: 1 KiB passed while
32 KiB timed out. Its `eth0` MTU was 1280. Restoring that underlying interface to
1500 allowed the 32 KiB transfer and an exact-byte 1 MiB transfer to complete.
The setting was recorded using WSL's native per-distribution boot configuration;
an operating-system restart was not part of this test. This was machine setup,
not a custom remote service or a Studio networking workaround.

Graphviz was absent, so the optional process-graph SVG was skipped. Reconstruction
continued. Gaussian splatting and physical multi-host scaling were not tested.

## Observed Modal Starter attempt

The authenticated workspace was visibly on Starter and had no apps before the
test. Its default image builder was `2026.10`.

The first real deployment exposed an SDK mismatch: locked Modal 1.5.4 did not
support that builder. Raising the dependency to 1.6.3 and updating the lockfile
allowed both workspace images to build through the Studio deployment UI.

Modal then rejected GPU function deployment with:

> Please add a payment method to use L4 GPU functions.

The user chose to keep the account without adding billing information. No plan
upgrade or payment method was added. All three test apps were subsequently
observed stopped with zero tasks; billing reported zero billed cost.

**Modal GPU reconstruction, cancellation, result retrieval, and GPU scale-down
remain unverified.** Image-build success is not cloud GPU execution evidence.

## Local checks

The follow-up passed 97 focused Python tests, 19 viewer tests, 12 production-bundle
browser tests, frontend type checking/build, Black checks for the changed Python
files, and the exact repository CI lint command. Raw run artifacts are retained
separately from the implementation diff.
