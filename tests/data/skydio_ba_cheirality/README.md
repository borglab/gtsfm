# Frozen BA cheirality regression

`scene.npz` contains 32 Cal3Bundler cameras, 3,330 points and 14,461 pixel
observations from the post-merge BA input in
[CPU capture trial 1](https://github.com/borglab/gtsfm/actions/runs/42281969753).
The source was `91f5b74f0c098108716599d449d8db84987c6e9b`, GTSAM 4.3.0,
and the public Skydio-32 CI dataset. It contains numeric scene data, not images,
credentials, Python pickles or executable objects.

The retained native input was cyclically relabeled by one track. This changes
variable/factor ordering while preserving all cameras, points and observations.
Both labelings have initial objective 2295.0549265074. With identity damping, the
relabeled input drops from 32 cameras/3,330 tracks to 9 cameras/4 tracks after the
normal 3-pixel measurement filter. Only 12 of 14,461 observations remain in front
of their camera, despite a reported final objective near zero. GTSAM's
[GeneralSFMFactor2 cheirality path](https://github.com/borglab/gtsam/blob/4.3.0/gtsam/slam/GeneralSFMFactor.h#L339)
returns a zero residual and zero Jacobians for behind-camera observations.

The regression calls the production `run_simple_ba` with GMC noise,
`measurement_noise_sigma=2.0` and no calibration prior, matching the captured
optimizer. Other constructor options retain their defaults. It checks both the
frozen labeling and a cyclic relabeling, requiring all 32 cameras and over 3,000
tracks to survive filtering. This is a collapse regression, not a general
accuracy or convergence guarantee. No seed was selected to obtain a passing run.

Stock GTSAM diagonal damping preserves 32 cameras and 3,324 filtered tracks on
the frozen input. Merely changing the linear solver to QR still collapses.
The diagonal setting scales the LM damping by the Hessian diagonal for the
different units of pose, landmark and calibration variables. It does not change
the reprojection objective or the CUDA solver. Full benchmark quality and
runtime still require validation.

Array schema (`allow_pickle=False`):

| Array | Shape | Meaning |
| --- | --- | --- |
| `camera_ids` | 32 | Original image indices |
| `poses` | 32 × 4 × 4 | Camera-to-world matrices |
| `calibrations` | 32 × 5 | `fx, k1, k2, px, py` |
| `points` | 3,330 × 3 | Track positions in frozen variable order |
| `offsets` | 3,331 | Measurement offset for each track |
| `measurement_camera_ids` | 14,461 | Image index of each observation |
| `measurements` | 14,461 × 2 | Pixel coordinates |

SHA-256: `50bce3d5202ca62a8ef9477da3df4c3d3f50bca9bd72f866290a3d976c8eeede`
