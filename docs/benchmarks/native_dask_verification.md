# Native Dask configuration verification

This change reuses Hydra/OmegaConf to construct Dask LocalCluster and native Dask
YAML/environment settings for runtime policies. It does not change the SfM algorithm.

The tested runner SHA256 is
`9a76568399ad840ee8ac51fa79eae297ba51c42c9aec763aa83e91b0e1515701`.

- 62 focused tests and frontend type check/build passed.
- Independent GPT-6-Astra verification passed 14 configuration/lifecycle cases on
  Dask/distributed 2025.9.1, including real worker settings and YAML/env precedence.
- Four authenticated HTTP-to-RTX-3080 reconstructions (manual/native/manual/native)
  produced byte-identical complete camera/image/point exports and named pairs:
  four cameras and 1,509 tracks each. Source, models, inputs and seeds were controlled.

This establishes output parity on the four-image fixture, without a speed, accuracy
or large-scene claim. Browser tests intercepted requests; remote routing checks
used stubs. Detailed limitations and retained failed probes are in the full report.

Experiment dumps and reproduction tools are archived separately from this PR's
file diff at immutable evidence revision `a6c6e014`:

- [One-page visual brief](https://github.com/borglab/gtsfm/blob/a6c6e0145d18c3001dedc65294985027fa408a5e/output/pdf/dask-profiles-verification.pdf)
- [Full independent report and reproduction instructions](https://github.com/borglab/gtsfm/blob/a6c6e0145d18c3001dedc65294985027fa408a5e/docs/benchmarks/native_dask_verification.md)
- [Raw evidence, manifests and complete exports](https://github.com/borglab/gtsfm/blob/a6c6e0145d18c3001dedc65294985027fa408a5e/docs/benchmarks/raw/README.md)
