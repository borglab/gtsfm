# Merging reconstructed cameras without running bundle adjustment again

[`camera_tree_optimizer.py`](camera_tree_optimizer.py) combines several small
reconstructions into one set of camera poses. The preferred producer,
`compute_camera_marginal_factors()`, accepts the original nonlinear BA factor
graph and estimated values, including the original gauge constraints and weights.
It returns a compact camera-only Gaussian graph. A separate camera-only optimizer
consumes these summaries and optimizes cameras and coordinate transformations
between leaves.

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
in camera poses and point locations have approximately linear effects on the
image discrepancies. Building this local approximation is called *linearization*.
The graph-based producer builds this approximation once at the supplied values,
using the original factors, noise models, and priors. It retains discrepancies at
those estimates; it does not assume the input is already a perfect fit. Camera
variables are six-dimensional `Pose3` values. Landmarks and any other non-camera
variables, such as variable calibration parameters, are marginalized out. The
output camera poses are subsequently merged with fixed calibration metadata.

Conceptually, the resulting least-squares problem is

```text
minimize || A_camera * camera_change + A_point * point_change - b ||².
```

For any proposed camera changes, we can solve for the best point changes. After
doing that algebraically, the remaining cost depends only on camera changes.
This is *landmark elimination*, often described using the Schur complement. The
preferred producer uses GTSAM's ordering and multifrontal QR APIs:

1. `Ordering.ColamdConstrainedLastGaussianFactorGraph()` places every non-camera
   variable before cameras. Partial multifrontal QR eliminates that prefix.
2. `Ordering.MetisGaussianFactorGraph()` computes nested dissection on the reduced
   camera graph, including the camera connections created by landmark elimination.
3. Camera multifrontal QR uses that ordering to produce a Gaussian Bayes tree.
   Its conditionals become the returned camera factor graph.

This avoids explicitly forming a dense Schur-complement matrix. The returned
graph has at most one factor per camera, usually fewer because a Bayes-tree clique
can contain several frontal cameras. It does not retain one factor per landmark.
Individual factors use dense blocks for their participating cameras; cameras
outside a factor's scope are omitted. Nested dissection reduces elimination
fill-in but cannot guarantee small cliques for inherently dense camera connectivity.
In a test with 12 cameras and tracks spanning neighboring triples, both 100 and
1,000 landmarks yield 7 camera factors, each involving at most 5 cameras.

This step retains the camera information in the linearized problem, up to an
additive constant that does not affect optimization. It also retains correlations:
the evidence about one camera can depend on how another camera moves. Replacing
the result with independent camera priors would generally lose those relations.

GTSAM's Python `Marginals` interface provides joint information and covariance
matrices, but those matrices alone do not retain the residual offset `b`. For
example, the costs `x²` and `(x - 3)²` have the same information matrix but different
minima. The installed Python `Marginals` binding exposes neither an ordered constructor nor
its internal Bayes tree or a joint marginal factor graph. The producer therefore
uses the underlying multifrontal QR machinery directly, retaining both the
quadratic and linear terms and any constrained noise models.

The graph-based producer requires the original graph to include its coordinate
gauge and preserves those constraints. When constructing `CameraLeafFactors`,
mark its output `gauge_fixed=True`, so the consumer does not add another gauge.
The geometry-based convenience producer instead returns an ungauged graph;
for those inputs the consumer chooses local coordinates. It constrains all six
coordinates of the first camera and one baseline-length direction toward the farthest camera. These seven
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
inputs must have exactly seven unobservable directions. Rank checks allow for
numerical noise from rounded rotations and Hessian square roots; additional
geometric degeneracy or an incorrect gauge declaration is rejected.

## Letting a local camera summary move and change scale

There is one global pose variable for each distinct camera, and one similarity
variable for each leaf. Internally, the leaf similarity maps **global coordinates
into the leaf frame**.

To evaluate a leaf's evidence, the solver:

1. Transforms each relevant global camera pose into that leaf's frame.
2. Measures its six-dimensional pose change from the leaf's estimated camera pose.
3. Substitutes those changes into the stored camera conditional.

In notation, a conditional's residual is

```text
delta_i = Local(estimated_camera_pose_i, S_leaf.transformFrom(global_pose_i))
residual = A * concatenate(delta_i) - b.
```

Here `Local` means a small rotation-and-translation displacement between poses,
not subtraction of their matrix entries. `A` and `b` come from the Gaussian
conditional. GTSAM's `Similarity3.transformFrom()` applies its scale convention
and provides derivatives. The implementation chains these derivatives with those
of `Pose3.localCoordinates()` in a `CustomFactor`.

Each factor keeps the conditional's noise model, including exact coordinate
constraints. Its residual is not whitened a second time. A `LinearContainerFactor`
alone would not supply the similarity-dependent conversion of global poses into
leaf coordinates.

## Assembly and the global solve

Initialization starts in the first leaf's coordinate frame. The solver traverses
the graph of overlapping leaves deterministically and aligns camera poses to
initialize the remaining similarities and cameras. Each usable overlap must
contain at least two shared cameras with distinct centers. This requirement uses
camera orientations as well as centers; it is not point-only alignment. The usable
overlap graph must connect all leaves.

The scene tree is then traversed bottom-up with `Tree.map_with_children()`.
Leaves contribute their condensed factors. Each internal node combines its
children's factors exactly once. There is no optimization at internal nodes:
one Levenberg–Marquardt optimization runs at the root, using multifrontal QR.

The first leaf's similarity is fixed to identity to choose the output origin,
orientation, and scale. All other similarities and all global camera poses are
optimized jointly. Afterward, the solver linearizes this joint graph, eliminates
the similarity variables, and factors the remaining camera graph into the final
Gaussian Bayes tree. That final tree describes the local camera quadratic in the
chosen global gauge; it contains neither points nor similarity variables.

## Preferred API: use the original BA graph and values

Immediately after leaf BA, while the nonlinear graph and optimized values are
still available:

```python
from gtsam import Values
from gtsam.symbol_shorthand import X

from gtsfm.camera_tree_optimizer import (
    CameraLeafFactors,
    LeafCameraMetadata,
    compute_camera_marginal_factors,
)

# Defaults to retaining the original graph's X(i) Pose3 camera variables.
camera_graph = compute_camera_marginal_factors(ba_graph, optimized_values)

estimated_camera_poses = Values()
camera_metadata = {}
for i, image in enumerate(images):
    estimated_camera_poses.insert(X(i), optimized_values.atPose3(X(i)))
    camera_metadata[X(i)] = LeafCameraMetadata(
        filename=image.filename,
        calibration=image.calibration,
        shape=image.shape,
    )
leaf_factors = CameraLeafFactors(
    camera_graph, estimated_camera_poses, camera_metadata, gauge_fixed=True
)
```

Pass `camera_keys=[...]` explicitly if camera variables use other keys. The
selected values must be `Pose3`, and the supplied graph must contain a complete
coordinate gauge. No gauge is added, removed, or replaced, and no BA runs in this
call. It preserves whatever priors are present; callers must distinguish intended
coordinate constraints from geometric priors. Other graph variables are eliminated
before cameras, and caller-owned inputs remain unchanged. METIS support in GTSAM
is required; there is no silent fallback to another ordering.

The graph and values must describe the original likelihood you want to preserve.
For example, an optimizer that keeps additional robust weights outside the graph
must incorporate those weights into the supplied graph. The producer does not
retrieve hidden optimizer state. `measurement_sigma` does not affect this API.

Place these leaf summaries into a `CameraFactorTree` and call
`optimize_camera_factor_tree()` as shown below. The graph producer itself needs no
`GtsfmData` or scene tree. Metadata is needed only when assembling leaves for the
camera-tree optimizer.

## Compatibility API: use reconstructed leaves

> **Warning:** `GtsfmData` stores reconstructed geometry and observations, but does
> not retain the original BA graph, gauge factors, or noise models. Prefer
> `compute_camera_marginal_factors()` with the original graph and values, followed
> by `optimize_camera_factor_tree()`. The SceneTree convenience wrapper emits a
> `UserWarning` with this guidance and remains available.

Given a `SceneTree` whose leaves contain named camera reconstructions, the
convenience wrapper computes each leaf Schur complement and delegates to the
camera-only optimizer. It ignores internal reconstruction payloads:

```python
from gtsfm.camera_tree_optimizer import CameraTreeOptions, optimize_camera_tree

result = optimize_camera_tree(scene_tree, CameraTreeOptions(measurement_sigma=1.0))

camera_scene = result.scene
leaf_transforms = result.leaf_to_global
camera_bayes_tree = result.camera_bayes_tree
print(result.initial_error, result.final_error)
```

## Producing and consuming camera factors separately

If only a saved reconstruction is available, the compatibility producer can
rebuild a measurement model:

```python
from gtsfm.camera_tree_optimizer import compute_camera_schur_complement

leaf_factors = compute_camera_schur_complement(leaf_reconstruction)
```

This call linearizes the supplied reconstruction once and eliminates its
landmarks with QR using one-pixel isotropic noise by default. It runs no BA and
adds no coordinate constraints. Unlike the preferred graph producer, it returns
the uncompressed reduced graph, potentially one factor per landmark; the consumer
compresses it later. It cannot recover original priors or noise models.
The returned `CameraLeafFactors` contains:

- `camera_graph`: the reduced `GaussianFactorGraph`, including its linear term.
- `estimated_camera_poses`: `Values` containing the frozen `Pose3` estimates.
- `camera_metadata`: a dictionary of `LeafCameraMetadata` with filename, fixed
  `Cal3Bundler` calibration, and optional image `shape`.
- `gauge_fixed`: `False` for this producer.

All three camera containers use the same keys: the input reconstruction's camera
indices. Different leaves can reuse those keys, or use entirely different,
noncontiguous keys. Filenames establish identity between leaves.

Build a `CameraFactorTree` with the desired hierarchy and pass it to the consumer:

```python
from pathlib import Path

from gtsfm.camera_tree_optimizer import CameraFactorTree, optimize_camera_factor_tree
from gtsfm.utils.tree import Tree

# left_factors and right_factors are CameraLeafFactors produced above.
# Their usable overlap must include two cameras with distinct centers.
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
`Pose3` local coordinates at `estimated_camera_poses`, with six columns per camera
and the original measurement weights. Information or covariance matrices alone
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
tracks. Global camera indices follow sorted image filenames. Calibration and
metadata are taken from the first occurrence of each camera.

`result.leaf_to_global` maps leaf paths to transformations **from the input leaf
frame into the output frame**. These are the inverses of the internal optimization
variables. `result.factor_tree` preserves the input hierarchy and exposes the
assembled factors and leaf summaries. Each leaf summary includes its local-to-global
camera index mapping, estimated camera poses, camera Bayes tree, and lifted factors.
Those returned poses and factors use global `X(i)` keys, while the mapping retains
the original leaf-local keys. Objective values omit additive constants discarded
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

The tree optimizer supports `Cal3Bundler` camera metadata with fixed intrinsics.
The original-graph producer accepts Pose3 camera variables without requiring
calibration metadata. Tests check compact factor counts, sparse scopes, exact
and soft prior preservation, nonzero residual offsets, landmark-first ordering,
and METIS camera ordering with multifrontal QR at both stages.
It rejects missing camera identities, disconnected or insufficient overlaps,
degenerate baselines, and insufficient geometric rank. A caller must arrange for
leaf evidence not to duplicate measurements; the solver does not deduplicate
tracks across arbitrary input reconstructions.

The root solve optimizes the assembled, frozen approximation. It does not update
landmarks, relinearize the original image measurements, or promise the global
optimum of full nonlinear BA. Large departures from the supplied leaf geometry
can make that approximation less accurate. The final Bayes tree likewise
describes a local linearization, not all possible global reconstruction solutions.
