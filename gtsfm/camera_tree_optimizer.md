# Merging reconstructed cameras without running bundle adjustment again

[`camera_tree_optimizer.py`](camera_tree_optimizer.py) combines several small
reconstructions into one set of camera poses and intrinsics. A public
Schur-complement producer uses image measurements and 3D points to summarize what
each leaf tells us about its cameras. A separate camera-only optimizer consumes
these summaries and optimizes camera poses, intrinsics, and the coordinate
transformations between leaves. You can use the supplied producer or compute the
summaries elsewhere.

The inputs must already be reconstructed. This is a standalone solver; it is not
currently part of the production scene-merging pipeline.

## The problem: each reconstruction has its own ruler

A camera pose describes where a camera is and which way it faces. A reconstruction
also contains landmarks: 3D points seen in multiple images. Bundle adjustment
(BA) adjusts the camera poses and landmarks to make their predicted image
locations agree with the measured locations.

An image-based reconstruction does not, by itself, tell us which direction is
globally up, where the origin is, or how long a meter is. Rotating, translating,
and uniformly rescaling the whole reconstruction leaves the predicted images
unchanged. These seven freedoms are called its *similarity gauge*: three for
rotation, three for translation, and one for scale.

Consequently, two independently reconstructed groups can contain the same camera
but give it different numerical poses. A similarity transformation converts
between their coordinate frames. Simply aligning the groups is a useful starting
point, but their estimates can disagree because of measurement noise. The solver
lets cameras move to reconcile those disagreements according to the information
in each leaf.

## Two different kinds of tree

The input **scene tree** organizes reconstructions into groups. Its leaves contain
the small reconstructions we use. Internal nodes describe how those groups are
assembled. Their reconstruction payloads, if present, are ignored.

For example, the Lund Door test uses this hierarchy. The ranges denote image
names, not the locally assigned camera indices in a COLMAP model.

```text
root
├── C_1
│   ├── C_1_1: cameras 1–5
│   └── C_1_2: cameras 3–7
└── C_2
    ├── C_2_1: cameras 6–10
    └── C_2_2: cameras 8–12
```

Every camera to be returned must appear in a leaf. Cameras can appear in several
leaves; that overlap connects their coordinate frames. Identity is determined by
image filename, since different COLMAP reconstructions may number the same image
differently.

A **Bayes tree** serves a different purpose: it stores a factored system of
constraints. Each node, called a clique, groups variables that must be handled
together. It stores a conditional relation between some cameras and cameras
shared with its parent. Taken together, those conditionals represent the camera
quadratic described below. The Bayes tree is computed by eliminating variables;
it need not have the same shape as the scene tree. A camera can belong to an
internal Bayes-tree clique.

## Summarizing a leaf without keeping its points

A *factor* is one piece of evidence about some variables. A reprojection factor,
for instance, connects a camera and a landmark and measures the discrepancy
between a predicted and an observed image location. A *factor graph* is a
collection of such factors and the variables they connect.

Reprojection is nonlinear. Around the supplied estimates, however, small changes
in camera poses, intrinsics, and point locations have approximately linear effects
on the image discrepancies. Building this approximation is called *linearization*.
The solver builds it once per leaf, retaining camera poses and intrinsics as
variables and using one-pixel isotropic measurement noise by default. Set
`CameraTreeOptions(optimize_intrinsics=False)` to keep intrinsics fixed instead.
It retains the discrepancies at the supplied estimates; it does not assume the
input is already a perfect fit.

Conceptually, the resulting least-squares problem is

```text
minimize || A_pose * pose_change + A_calibration * calibration_change
            + A_point * point_change - b ||².
```

For any proposed camera changes, we can solve for the best point changes. After
doing that algebraically, the remaining cost depends only on camera changes.
This is *landmark elimination*, often described using the Schur complement. The
implementation uses `GaussianFactorGraph.marginal(camera_keys, EliminateQR)`:
GTSAM chooses an ordering and eliminates all variables other than the requested
camera poses and calibrations, without forming a Schur-complement matrix.
Only landmarks are eliminated. Pose–intrinsics and cross-camera correlations
remain in the reduced graph; intrinsics are neither fixed nor marginalized in
the default mode.

This step retains the camera information in the linearized problem, up to an
additive constant that does not affect optimization. It also retains correlations:
the evidence about one camera can depend on how another camera moves. Replacing
the result with independent camera priors would generally lose those relations.

GTSAM's Python `Marginals` interface provides joint information and covariance
matrices, but those matrices alone do not retain the residual offset `b`. For
example, the costs `x²` and `(x - 3)²` have the same information matrix but different
minima. The factor-graph `marginal()` method retains both the quadratic and linear
terms, as well as any constrained noise models, which the merging solver needs.

The public producer returns this camera graph without adding coordinate
constraints (`gauge_fixed=False`). Before factoring it into a Gaussian Bayes tree,
the camera-only consumer chooses local coordinates. It constrains all six coordinates of the first
camera and one baseline-length direction toward the farthest camera. These seven
exact constraints remove the local similarity gauge. GTSAM's `NonlinearEqualityPose3`
and a constrained `RangeFactorPose3` supply these rows by linearizing once at the
estimated camera poses. GTSAM also chooses the COLAMD ordering for the subsequent
QR factorization. The summary then contains
camera conditionals and estimated camera poses, with no landmark variables. These
poses stay fixed at the supplied leaf estimates throughout the root solve. A separate
similarity variable will let this entire local coordinate system move globally.

External producers may instead supply a graph with its coordinate gauge already
fixed and set `gauge_fixed=True`. The consumer requires a full-rank system in
that case and preserves those constraints without adding another gauge. Ungauged
inputs must have exactly seven unobservable directions. For N cameras, the
expected rank is `9*N - 7` with variable intrinsics and `6*N - 7` with fixed
intrinsics. Self-calibration can introduce additional degeneracies even when
fixed-intrinsics poses are observable; these are rejected rather than hidden
with calibration priors. Rank checks allow for numerical noise from rounded
rotations and Hessian square roots; additional
geometric degeneracy or an incorrect gauge declaration is rejected.

## Letting a local camera summary move and change scale

There is one global pose and one calibration variable for each distinct image,
and one similarity variable for each leaf. Images are identified by filename.
Different images have independent calibrations; occurrences of the same image
share one global calibration. Each leaf retains its own frozen calibration
estimate. In fixed-intrinsics mode there are no calibration variables. Internally,
the leaf similarity maps **global coordinates into the leaf frame**.

To evaluate a leaf's evidence, the solver:

1. Transforms each relevant global camera pose into that leaf's frame.
2. Measures its six-dimensional pose change from the leaf's estimated camera pose.
3. Measures each calibration change from that leaf’s frozen calibration estimate.
4. Substitutes both kinds of changes into the stored camera conditional.

In notation, a conditional's residual is

```text
pose_delta_i = Local(estimated_camera_pose_i, S_leaf.transformFrom(global_pose_i))
calibration_delta_i = Local(estimated_calibration_i, global_calibration_i)
residual = A * concatenate(deltas_in_factor_key_order) - b.
```

Here `Local` means a small rotation-and-translation displacement between poses,
not subtraction of their matrix entries. `A` and `b` come from the Gaussian
conditional. GTSAM's `Similarity3.transformFrom()` applies its scale convention
and provides derivatives. The implementation chains these derivatives with those
of `Pose3.localCoordinates()` in a `CustomFactor`. Calibrations use
`Cal3Bundler.localCoordinates()`: their three coordinates are focal length and
the two radial-distortion coefficients, `f`, `k1`, and `k2`. Their derivative
is identity and they are unaffected by similarities. Principal points stay fixed;
variable-intrinsics leaves must agree on the principal point for a shared image.

Each factor keeps the conditional's noise model, including exact coordinate
constraints. Its residual is not whitened a second time. A `LinearContainerFactor`
alone would not supply the similarity-dependent conversion of global poses into
leaf coordinates.

## Assembly and the global solve

Initialization starts in the first leaf's coordinate frame. The solver traverses
the graph of overlapping leaves deterministically and aligns camera poses to
initialize the remaining similarities and cameras. Each global calibration is
initialized from the first occurrence of its image in tree traversal order. Each
usable overlap must contain at least two shared cameras with distinct centers. This requirement uses
camera orientations as well as centers; it is not point-only alignment. The usable
overlap graph must connect all leaves.

The scene tree is then traversed bottom-up with `Tree.map_with_children()`.
Leaves contribute their condensed factors. Each internal node combines its
children's factors exactly once. There is no optimization at internal nodes:
one Levenberg–Marquardt optimization runs at the root, using multifrontal QR.

The first leaf's similarity is fixed to identity to choose the output origin,
orientation, and scale. All other similarities, global camera poses, and variable
intrinsics are optimized jointly. Afterward, the solver linearizes this joint graph, eliminates
the similarity variables, and factors the remaining camera graph into the final
Gaussian Bayes tree. That final tree describes the local camera quadratic in the
chosen global gauge; it retains `X(i)` poses and `K(i)` calibrations, but contains
neither points nor similarity variables. In fixed-intrinsics mode it retains
only `X(i)` poses. The output cameras contain the optimized intrinsics.

## Using reconstructed leaves

Given a `SceneTree` whose leaves contain named camera reconstructions, the
convenience wrapper computes each leaf Schur complement and delegates to the
camera-only optimizer. It ignores internal reconstruction payloads:

```python
from gtsfm.camera_tree_optimizer import CameraTreeOptions, optimize_camera_tree

result = optimize_camera_tree(scene_tree, CameraTreeOptions(measurement_sigma=1.0))

# Reproduce the original pose-only behavior when intrinsics should stay fixed:
fixed_result = optimize_camera_tree(scene_tree, CameraTreeOptions(optimize_intrinsics=False))

camera_scene = result.scene
leaf_transforms = result.leaf_to_global
camera_bayes_tree = result.camera_bayes_tree
print(result.initial_error, result.final_error)
```

## Producing and consuming camera factors separately

To compute a leaf summary independently:

```python
from gtsfm.camera_tree_optimizer import compute_camera_schur_complement

leaf_factors = compute_camera_schur_complement(leaf_reconstruction)
```

This call linearizes the supplied reconstruction once and eliminates its
landmarks with QR. It runs no BA and adds no coordinate constraints. The returned
`CameraLeafFactors` contains:

- `camera_graph`: the reduced `GaussianFactorGraph`, including its linear term
  and both pose and calibration variables in the default mode.
- `estimated_camera_poses`: `Values` containing the frozen `Pose3` estimates.
- `camera_metadata`: a dictionary of `LeafCameraMetadata` with filename, frozen
  `Cal3Bundler` estimate, and optional image `shape`.
- `calibration_keys`: a mapping from every pose key to its calibration key, or
  an empty mapping for fixed intrinsics. The producer uses input indices for
  poses and `K(i)` for calibrations.
- `gauge_fixed`: `False` for this producer.

Pose estimates and metadata use the input reconstruction’s camera indices.
Calibration keys must be distinct from each other and from the pose keys within
a leaf, and the graph keys must exactly match the declared variables. External
producers may use arbitrary, noncontiguous keys, reused across leaves; key symbols
do not determine variable types. Filenames establish identity between leaves.

The consumer infers the variable layout from the summaries: `optimize_intrinsics`
and `measurement_sigma` are producer-only options and do not reinterpret or
reweight supplied factors. Existing `CameraLeafFactors` construction remains
valid because `calibration_keys` defaults to an empty mapping. Entire pose-only
trees remain supported, but mixing pose-only and variable-intrinsics leaves in
one tree is rejected.

Build a `CameraFactorTree` with the desired hierarchy and pass it to the consumer:

```python
from pathlib import Path

from gtsfm.camera_tree_optimizer import CameraFactorTree, optimize_camera_factor_tree
from gtsfm.utils.tree import Tree

# These leaves must share at least two cameras with distinct centers.
left_factors = compute_camera_schur_complement(left_reconstruction)
right_factors = compute_camera_schur_complement(right_reconstruction)
camera_factor_tree: CameraFactorTree = Tree(
    (Path("root"), None),
    children=(
        Tree((Path("root/left"), left_factors)),
        Tree((Path("root/right"), right_factors)),
    ),
)
result = optimize_camera_factor_tree(camera_factor_tree)
```

An external producer can construct `CameraLeafFactors` directly with these same
fields. Its factors must describe the complete reduced quadratic in GTSAM
`Pose3` local coordinates at `estimated_camera_poses`, with six columns per pose,
and optionally `Cal3Bundler` local coordinates at the metadata calibration, with
three columns per calibration key. The original measurement weights must be
preserved. Information or covariance matrices alone
are insufficient: the linear term is needed too. Jacobian and Hessian factors
are accepted. Producers setting `gauge_fixed=True` must supply coordinate-fixing
constraints, rather than geometric priors that unintentionally change the model.
Mixed trees of gauged and ungauged leaves are supported.

The consumer never reads tracks, evaluates image observations, or eliminates
landmarks. `measurement_sigma` affects only `compute_camera_schur_complement()`;
it does not reweight externally supplied factors. The consumer copies and rekeys
factors with `LinearContainerFactor.rekey()`, preserving column associations,
linear terms, and noise models without modifying caller-owned graphs or estimates.
Invalid leaf inputs are reported with their tree paths.

Both workflows preserve the input hierarchy and produce the same result for the
same camera quadratics. This is an in-memory Python API; serialization and
distributed execution are outside its scope.

## Understanding the result

`result.scene` is a `GtsfmData` containing cameras and image metadata, with zero
tracks. Global camera indices follow sorted image filenames. Calibrations are
optimized in the default mode; fixed-intrinsics mode takes them from the first
occurrence of each camera. Image metadata always comes from that first occurrence.

`result.leaf_to_global` maps leaf paths to transformations **from the input leaf
frame into the output frame**. These are the inverses of the internal optimization
variables. `result.factor_tree` preserves the input hierarchy and exposes the
assembled factors and leaf summaries. Each leaf summary includes its local-to-global
camera index mapping, estimated camera poses, `estimated_calibrations`, camera
Bayes tree, and lifted factors.
Returned poses use global `X(i)` keys and calibrations use `K(i)` keys; factors
may contain both. The mapping retains the original leaf-local keys. Objective
values omit additive constants discarded
during elimination, just as in the reconstruction wrapper.

## What the tests establish, and what the approximation means

The Lund test preserves the existing seven-node binary topology but constructs
overlapping leaves in memory from the full reconstruction. Each source track is
assigned to exactly one leaf, choosing the leaf with the most observations and
breaking ties deterministically using its track index. Only that leaf's
observations are retained, with at least two required. Thus a source measurement
is not counted in multiple leaf likelihoods. The fixture files stay unchanged.

The tests check equivalence with explicit landmark minimization, preservation of
the constrained camera quadratic, analytic derivatives against finite differences,
recursive assembly, recovery of all 12 cameras, and invariance under independent
leaf rotations, translations, and scales. They also check invalid inputs and
verify that no BA optimizer runs and internal reconstructions are ignored. Public
producer/consumer execution is compared with the reconstruction wrapper, including
objectives and the final camera quadratic. External-input tests cover mixed gauges,
reused local keys, immutable inputs, and execution without reconstruction access.
Both intrinsics modes run through the Lund workflows. Additional tests profile
out points in the joint pose/intrinsics system, check mixed-block and
calibration-only derivatives and weighting, reconcile independent calibration
estimates, and reject invalid calibration mappings and unobservable intrinsics.

The implementation supports `Cal3Bundler` cameras with variable intrinsics by
default, with an explicit fixed-intrinsics option. Shared intrinsics across
different images and production scene-merging integration are outside this API.
It rejects missing camera identities, disconnected or insufficient overlaps,
degenerate baselines, and insufficient geometric rank. A caller must arrange for
leaf evidence not to duplicate measurements; the solver does not deduplicate
tracks across arbitrary input reconstructions.

The root solve optimizes the assembled, frozen approximation. It does not update
landmarks, relinearize the original image measurements, or promise the global
optimum of full nonlinear BA. Large departures from the supplied leaf geometry
can make that approximation less accurate. The final Bayes tree likewise
describes a local linearization, not all possible global reconstruction solutions.
