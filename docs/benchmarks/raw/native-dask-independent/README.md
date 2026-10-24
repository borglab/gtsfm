# Independent native Dask evidence

Read [the verification report](../../native_dask_verification.md) for the scoped verdicts.

- `acceptance.json`, `acceptance-original.md`, `method-addendum.md`: pre-output acceptance and approved transport refinement; thresholds unchanged.
- `final-cpu-1/`: final frozen-source 14-case real-worker, failure-cleanup, ASGI and stub-scoped legacy matrix, all VERIFIED.
- `final-source-audit.json`, `documentation-after.json`, `netdiff-skill-review.json`: source removal, repaired guide and prefeature/canonical-skill checks.
- `gpu-preflight-review.json`: frozen source/input/model preflight review, not a runtime verdict.
- `gpu-four-trial-comparison.json`: independently executed unchanged strict comparator; complete new final exports and pairs byte-identical, all comparisons VERIFIED.
- `gpu-controls-review.json`: independent source/assets/config/cache/seeds/actual-worker/HTTP/cleanup review; VERIFIED within its stated scope.
- `mira-lifecycle-review.json`: original strict epoch identity failure retained, post-run stable kernel evidence, whole-series unrelated service identity INCONCLUSIVE.
- `manifest.json`: hashes of this directory (excluding itself and ephemeral scratch lock files), the report and the frozen CPU harness.

Intermediate investigations remain separate: `runner-before/` failed native timeout precedence; `constructor-interim-1/` exposed intermediate timeout/legacy issues (one source-drift case is INCONCLUSIVE); `defaults-interim-1/` failed preservation of project defaults. `native-defaults-api-probe.json` retains a rejected merge proposal, while `native-defaults-merge-probe.json` records the successful upstream API sequence. `studio-1/` is a verifier-input error, not a production regression; `studio-2/` is the corrected routing probe. `upstream-1/` tests upstream Dask only.

GPU artifacts are in the sibling `native-dask-gpu/` directory and were produced by the executor, not this reviewer. This reviewer launched no GPU jobs, installed no dependencies, edited no production files and made no commits. The exact comparator command and compared-file hashes are embedded in its JSON result. Historical `gpu-experiment/` and the strict comparator remain unchanged. No speedup or internal tensor bitwise-equality claim is made.
