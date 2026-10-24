# Pre-output method addendum, 2026-10-24 23:05 UTC

Coordinator message `msg_456bfc14fcee`, received before any new GPU launch, approves
identical authenticated loopback HTTP `POST /api/jobs` transport for all four trials.
The comparison is manual constructor arguments (A1/A2) versus equivalent Hydra
LocalCluster YAML (B1/B2); B requests intentionally conflicting manual form defaults
that must be ignored when the constructor file is authoritative. Execution order remains
A1/B1/A2/B2; strict comparator argument order remains A1/A2/B1/B2.

This closes a launch-path confound and tests real HTTP dispatch in both arms. It changes
no output identity requirements, numerical tolerances, repeatability rules, cache/model/
data/seed/worker controls, failure classification, or no-speedup boundary. Direct runner
construction remains independently exercised in bounded CPU subprocesses. The original
acceptance snapshot and hash remain unchanged.

Independent pre-run harness review requested recording the executing harness SHA256 and
requiring the actual --dataset path to match the frozen dataset path (or an equivalent
explicit immutable hash check). No reconstruction output has been inspected for this
new series, and no GPU permission is inferred from this addendum.
