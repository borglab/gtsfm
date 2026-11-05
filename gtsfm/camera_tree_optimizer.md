# Merging reconstructed cameras without running bundle adjustment again

[`camera_tree_optimizer.py`](camera_tree_optimizer.py) combines several small
reconstructions into one set of camera poses. It uses the image measurements and
3D points in each input leaf to summarize what that leaf tells us about its
cameras. It then discards the points and optimizes only cameras and the coordinate
transformations between reconstructions.

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
The solver builds it once per leaf, keeping camera intrinsics fixed and using
one-pixel isotropic measurement noise by default. It retains the discrepancies
at the supplied estimates; it does not assume the input is already a perfect fit.

Conceptually, the resulting least-squares problem is

```text
minimize || A_camera * camera_change + A_point * point_change - b ||².
```

For any proposed camera changes, we can solve for the best point changes. After
doing that algebraically, the remaining cost depends only on camera changes.
This is *landmark elimination*, often described using the Schur complement. The
implementation uses GTSAM's multifrontal QR elimination instead of explicitly
forming a Schur-complement matrix.

This step retains the camera information in the linearized problem, up to an
additive constant that does not affect optimization. It also retains correlations:
the evidence about one camera can depend on how another camera moves. Replacing
the result with independent camera priors would generally lose those relations.

Before factoring the remaining camera graph into a Gaussian Bayes tree, the
solver chooses local coordinates. It constrains all six coordinates of the first
camera and one baseline-length direction toward the farthest camera. These seven
exact constraints remove the local similarity gauge. The summary then contains
camera conditionals and reference poses, with no landmark variables. A separate
similarity variable will let this entire local coordinate system move globally.

## Letting a local camera summary move and change scale

There is one global pose variable for each distinct camera, and one similarity
variable for each leaf. Internally, the leaf similarity maps **global coordinates
into the leaf frame**.

To evaluate a leaf's evidence, the solver:

1. Transforms each relevant global camera pose into that leaf's frame.
2. Measures its six-dimensional pose change from the leaf's reference pose.
3. Substitutes those changes into the stored camera conditional.

In notation, a conditional's residual is

```text
delta_i = Local(reference_leaf_pose_i, S_leaf.transformFrom(global_pose_i))
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

## Using the API

Given a `SceneTree` whose leaves contain named camera reconstructions:

```python
from gtsfm.camera_tree_optimizer import CameraTreeOptions, optimize_camera_tree

result = optimize_camera_tree(scene_tree, CameraTreeOptions(measurement_sigma=1.0))

camera_scene = result.scene
leaf_transforms = result.leaf_to_global
camera_bayes_tree = result.camera_bayes_tree
print(result.initial_error, result.final_error)
```

`result.scene` is a `GtsfmData` containing cameras and image metadata, with zero
tracks. Global camera indices follow sorted image filenames. Calibration and
metadata are taken from the first occurrence of each camera.

`result.leaf_to_global` maps leaf paths to transformations **from the input leaf
frame into the output frame**. These are the inverses of the internal optimization
variables. `result.factor_tree` preserves the input hierarchy and exposes the
assembled factors and leaf summaries. Each leaf summary includes its local-to-global
camera index mapping, reference poses, camera Bayes tree, and lifted factors.

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
verify that no BA optimizer runs and internal reconstructions are ignored.

The current implementation supports `Cal3Bundler` cameras with fixed intrinsics.
It rejects missing camera identities, disconnected or insufficient overlaps,
degenerate baselines, and insufficient geometric rank. A caller must arrange for
leaf evidence not to duplicate measurements; the solver does not deduplicate
tracks across arbitrary input reconstructions.

The root solve optimizes the assembled, frozen approximation. It does not update
landmarks, relinearize the original image measurements, or promise the global
optimum of full nonlinear BA. Large departures from the supplied leaf geometry
can make that approximation less accurate. The final Bayes tree likewise
describes a local linearization, not all possible global reconstruction solutions.
