/* GTSFM Pipeline Visualizer — minimal Babylon.js viewer.
 *
 * M2: discrete stage stepping (load each stage, swap point cloud).
 * M3 will add: smooth interpolation between stages, wireframe frustums,
 *              image-RGB color sampling, transition cross-fades.
 *
 * Uses Babylon globals from CDN (BABYLON.*). No build step required. */

// ── Scene setup ─────────────────────────────────────────────────────────────
const canvas = document.getElementById("renderCanvas");
const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
const scene = new BABYLON.Scene(engine);
scene.clearColor = BABYLON.Color4.FromHexString("#f6f6f3ff"); // matches CSS bg

const camera = new BABYLON.ArcRotateCamera(
  "cam", -Math.PI / 2, Math.PI / 2.2, 15,
  new BABYLON.Vector3(0, 0, 0), scene,
);
camera.minZ = 0.01;
camera.maxZ = 10000;
// Default Babylon up (Y-up). We compute a per-scene up-alignment rotation that
// maps the data's natural up to +Y, then apply it to all positions at parse.
// (See compute_up_alignment in scripts/visualize_gp_convergence.py for the source.)

// Sensibilities: lower number = MORE sensitive in Babylon.
camera.angularSensibilityX = 600;   // horizontal drag → alpha (yaw)
camera.angularSensibilityY = 600;   // vertical drag → beta (pitch)
camera.panningSensibility = 200;    // right-mouse pan
camera.wheelPrecision = 8;
camera.lowerRadiusLimit = 0.001;
camera.upperRadiusLimit = 10000;
// Prevent pole-crossing flips. JS has no Math.TWO_PI (NaN locked the camera before).
camera.lowerBetaLimit = 0.05;
camera.upperBetaLimit = Math.PI - 0.05;
// Inertia low → predictable instant response (high inertia can feel like "drift").
camera.inertia = 0.5;
camera.panningInertia = 0.5;

// ── Inputs: clear defaults (some have touch/twist that interferes on macOS
// trackpads), add ONLY the ones we want, THEN attachControl to wire DOM events.
// Order is critical: inputs.clear() detaches everything; addX() registers new
// inputs but doesn't bind to canvas; attachControl(canvas) binds them.
camera.inputs.clear();
camera.inputs.addPointers();      // LMB-drag orbit, RMB-drag pan, touch
camera.inputs.addMouseWheel();    // wheel/trackpad-pinch zoom
camera.inputs.addKeyboard();      // arrow keys for continuous orbit
// Built-in arrow-key input keycodes.
camera.keysUp = [38];     // ArrowUp     → beta-
camera.keysDown = [40];   // ArrowDown   → beta+
camera.keysLeft = [37];   // ArrowLeft   → alpha-
camera.keysRight = [39];  // ArrowRight  → alpha+
// Attach now that inputs are fully configured.
camera.attachControl(canvas, true);

// Make canvas focusable + auto-focus on hover so keyboard events reach it
// (without this, arrow keys go to the body and Babylon never sees them).
canvas.tabIndex = 0;
canvas.style.outline = "none";  // hide focus ring
canvas.addEventListener("mouseenter", () => canvas.focus());

// ── Up-alignment + scene framing (ported from scripts/visualize_gp_convergence.py)
// Active scene transform: 3x3 rotation that maps data-frame → Babylon-frame (Y-up).
// Computed once per RUN (not per stage) so all stages share the same orientation.
let sceneRotation = null;     // 3x3 row-major; null → identity
let sceneViewCenter = null;   // [x,y,z] median of all positions in transformed frame
let sceneViewRange = null;    // 90th percentile distance from center in transformed frame

function applySceneRotation(x, y, z) {
  if (!sceneRotation) return [x, y, z];
  const R = sceneRotation;
  return [
    R[0][0]*x + R[0][1]*y + R[0][2]*z,
    R[1][0]*x + R[1][1]*y + R[1][2]*z,
    R[2][0]*x + R[2][1]*y + R[2][2]*z,
  ];
}

// Compute a rotation that aligns scene "up" with Babylon's +Y.
// Strategy mirrors compute_up_alignment in the matplotlib script:
//  1) Avg camera image-up in world (wRi @ [0,-1,0]); use if cameras agree (|avg|>0.5)
//  2) Else PCA least-variance axis over all positions
//  3) Flip if landmarks sit BELOW cameras under the proposed up
function computeUpAlignment(camPositions, camRotations, landmarks) {
  let sceneUp = null;

  if (camRotations && camRotations.length > 0) {
    // R from parseImages is cam_from_world (COLMAP convention). The world-space
    // "image up" vector is wRi @ [0,-1,0] = (cam_from_world)^T @ [0,-1,0]
    //   = -row1(cam_from_world) = (-R[1][0], -R[1][1], -R[1][2]).
    // (Earlier this code used -col1(R) which gave wrong direction → upside-down scene.)
    let sx = 0, sy = 0, sz = 0;
    for (const R of camRotations) {
      sx += -R[1][0]; sy += -R[1][1]; sz += -R[1][2];
    }
    const n = camRotations.length;
    sx /= n; sy /= n; sz /= n;
    const mag = Math.hypot(sx, sy, sz);
    if (mag > 0.5) sceneUp = [sx / mag, sy / mag, sz / mag];
  }

  if (!sceneUp) {
    // PCA least-variance axis as fallback. Use power iteration on the
    // smallest eigenvalue of the covariance matrix (cheap, no SVD library).
    const all = landmarks && landmarks.length > 0 ? camPositions.concat(landmarks) : camPositions;
    if (all.length < 3) return null;
    let mx = 0, my = 0, mz = 0;
    for (const [x, y, z] of all) { mx += x; my += y; mz += z; }
    mx /= all.length; my /= all.length; mz /= all.length;
    let cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
    for (const [x, y, z] of all) {
      const dx = x - mx, dy = y - my, dz = z - mz;
      cxx += dx*dx; cyy += dy*dy; czz += dz*dz;
      cxy += dx*dy; cxz += dx*dz; cyz += dy*dz;
    }
    // Smallest eigenvector via inverse power iteration on (C - μI)⁻¹.
    // For our purposes: pick the world axis with smallest variance as a hint.
    if (czz <= cxx && czz <= cyy) sceneUp = [0, 0, 1];
    else if (cyy <= cxx) sceneUp = [0, 1, 0];
    else sceneUp = [1, 0, 0];
  }

  // Heuristic: landmarks should sit above cameras under the proposed up.
  if (landmarks && landmarks.length > 0 && camPositions.length > 0) {
    let lcx=0, lcy=0, lcz=0; for (const [x,y,z] of landmarks) {lcx+=x; lcy+=y; lcz+=z;}
    lcx /= landmarks.length; lcy /= landmarks.length; lcz /= landmarks.length;
    let ccx=0, ccy=0, ccz=0; for (const [x,y,z] of camPositions) {ccx+=x; ccy+=y; ccz+=z;}
    ccx /= camPositions.length; ccy /= camPositions.length; ccz /= camPositions.length;
    const dot = (lcx-ccx)*sceneUp[0] + (lcy-ccy)*sceneUp[1] + (lcz-ccz)*sceneUp[2];
    if (dot < 0) sceneUp = sceneUp.map(v => -v);
  }

  // Build rotation R mapping sceneUp → +Y. Pick a stable reference axis,
  // then construct an orthonormal basis [right, up, fwd] with up = sceneUp.
  let ref = [1, 0, 0];
  if (Math.abs(ref[0]*sceneUp[0] + ref[1]*sceneUp[1] + ref[2]*sceneUp[2]) > 0.95) ref = [0, 1, 0];
  // x_axis = sceneUp × ref (orthogonal to up)
  const xa = [
    sceneUp[1]*ref[2] - sceneUp[2]*ref[1],
    sceneUp[2]*ref[0] - sceneUp[0]*ref[2],
    sceneUp[0]*ref[1] - sceneUp[1]*ref[0],
  ];
  const xn = Math.hypot(...xa); for (let i = 0; i < 3; i++) xa[i] /= (xn + 1e-12);
  // z_axis = sceneUp × xa
  const za = [
    sceneUp[1]*xa[2] - sceneUp[2]*xa[1],
    sceneUp[2]*xa[0] - sceneUp[0]*xa[2],
    sceneUp[0]*xa[1] - sceneUp[1]*xa[0],
  ];
  // R rows: [x_axis, sceneUp, z_axis] — input p maps to (xa·p, sceneUp·p, za·p),
  // so the second component (Y) becomes sceneUp · p ⇒ +Y is the data's up.
  return [
    [xa[0], xa[1], xa[2]],
    [sceneUp[0], sceneUp[1], sceneUp[2]],
    [za[0], za[1], za[2]],
  ];
}

// View center = median of positions in transformed frame; range = 90th percentile
// distance from center × 1.05. Robust to outlier landmarks that would otherwise
// blow up a bounding-box fit.
function computeViewSetup(transformedPoints) {
  if (transformedPoints.length === 0) return null;
  // Per-axis median.
  const xs = transformedPoints.map(p => p[0]).sort((a, b) => a - b);
  const ys = transformedPoints.map(p => p[1]).sort((a, b) => a - b);
  const zs = transformedPoints.map(p => p[2]).sort((a, b) => a - b);
  const mid = arr => arr[Math.floor(arr.length / 2)];
  const center = [mid(xs), mid(ys), mid(zs)];
  // 90th-percentile distance from center.
  const dists = transformedPoints.map(p => Math.hypot(p[0]-center[0], p[1]-center[1], p[2]-center[2])).sort((a, b) => a - b);
  const range = dists[Math.floor(dists.length * 0.9)] * 1.05;
  return { center, range };
}

// Initial view azimuth: place camera on the SAME XZ-plane side as the camera
// cluster (i.e. where the photographers actually stood). This is more robust than
// the PCA-perpendicular approach — PCA returns one of two normals arbitrarily,
// which can land you behind/inside the structure on some scenes.
function computeAutoCameraAngles(transformedPoints, transformedCamPositions) {
  if (!transformedCamPositions || transformedCamPositions.length === 0 || transformedPoints.length < 3) {
    return { alpha: -Math.PI / 2, beta: Math.PI / 3 };
  }
  // Centroid of all points (≈ structure center) and centroid of cameras (≈ photographer side).
  let pmx = 0, pmz = 0;
  for (const [x, , z] of transformedPoints) { pmx += x; pmz += z; }
  pmx /= transformedPoints.length; pmz /= transformedPoints.length;
  let cmx = 0, cmz = 0;
  for (const [x, , z] of transformedCamPositions) { cmx += x; cmz += z; }
  cmx /= transformedCamPositions.length; cmz /= transformedCamPositions.length;
  // Direction from structure center to photographer cluster, in XZ plane.
  // Camera should sit on this side looking back at the structure.
  const vx = cmx - pmx, vz = cmz - pmz;
  const norm = Math.hypot(vx, vz);
  // Babylon ArcRotateCamera: alpha = atan2(z, x), camera position is at
  // (target.x + r*cos(α)*sin(β), target.y + r*cos(β), target.z + r*sin(α)*sin(β)).
  // We want camera toward (vx, vz) from target, so alpha = atan2(vz, vx).
  let alpha = norm > 1e-6 ? Math.atan2(vz, vx) : -Math.PI / 2;
  // Elev 30° above horizontal: Babylon beta from +Y axis → π/3.
  const beta = Math.PI / 3;
  return { alpha, beta };
}

new BABYLON.HemisphericLight("h", new BABYLON.Vector3(0, 1, 0), scene).intensity = 0.85;

engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());

// Camera-flip / reset hotkeys. Arrow keys are handled by Babylon's built-in
// addKeyboard input.
//   R → reset to initial auto-framing
//   F → flip front/back (alpha += π)
//   T → flip top/bottom by rotating the alignment 180° about its x-axis
//        (rebuilds rotation, re-applies it to the data, and re-frames).
//        A proper rotation (det = +1) — handy if up-alignment landed inverted.
let savedView = null;
function flipSceneUpAndReframe() {
  if (!sceneRotation || !currentRun) return;
  // Rotate the alignment 180° about its x-axis: negate the up (Y) AND forward (Z)
  // rows. Negating only the up row would set det = -1 — a MIRROR reflection that
  // renders the scene with reversed chirality (facades read backwards).
  sceneRotation[1] = sceneRotation[1].map(v => -v);
  sceneRotation[2] = sceneRotation[2].map(v => -v);
  // Reload current stage so re-parsing uses the updated rotation.
  loadStage(currentStageIdx, { animate: false });
  // Re-frame: transformed center's y and z invert with the flipped rows.
  if (savedView) {
    savedView.center[1] = -savedView.center[1];
    savedView.center[2] = -savedView.center[2];
    camera.target.set(savedView.center[0], savedView.center[1], savedView.center[2]);
  }
}
let uiHidden = false;
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === "r" || e.key === "R") {
    if (savedView) {
      camera.target.set(savedView.center[0], savedView.center[1], savedView.center[2]);
      camera.alpha = savedView.alpha;
      camera.beta = savedView.beta;
      camera.radius = savedView.radius;
    }
    e.preventDefault();
  } else if (e.key === "f" || e.key === "F") {
    camera.alpha += Math.PI;  // front ↔ back
    e.preventDefault();
  } else if (e.key === "t" || e.key === "T") {
    flipSceneUpAndReframe();   // top ↔ bottom (true vertical flip via scene-up flip)
    e.preventDefault();
  } else if (e.key === "h" || e.key === "H") {
    // Hide ALL overlays for a clean money shot; press again to restore.
    uiHidden = !uiHidden;
    for (const id of ["statsCard", "helpCard", "controlBar"]) {
      const el = document.getElementById(id);
      if (el) el.style.display = uiHidden ? "none" : "";
    }
    e.preventDefault();
  }
});

// ── State ───────────────────────────────────────────────────────────────────
let currentRun = null;        // { id, manifest_url, data_url_prefix, manifest }
let currentStageIdx = 0;
let pointsMesh = null;
let frustumLines = null;       // LinesSystem for core / untiered frustums
let frustumLinesAnnex = null;  // LinesSystem for flagged (zero-observation) frustums, drawn red
let isPlaying = false;
let playTimer = null;
let prevStageData = null;      // previous stage points (for points LERP)
let prevStageImages = null;    // previous stage cameras (for frustum LERP)
let lerpAnimRAF = null;        // requestAnimationFrame handle for points lerp
let frustumLerpRAF = null;     // requestAnimationFrame handle for frustum lerp
const LERP_MS = 600;           // ms to interpolate between consecutive stages
let pointSize = 2.0;           // current point cloud size, controlled by UI slider
let frustumScale = 0.015;      // frustum size as a fraction of viewRange, controlled by UI slider

// ── COLMAP parsing (minimal — just what we need) ────────────────────────────
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} failed: ${r.status}`);
  return await r.text();
}

function parsePoints3D(text) {
  // Format per line: POINT3D_ID X Y Z R G B ERROR TRACK[]
  // Applies sceneRotation if set so all positions live in the aligned (Y-up) frame.
  const xyz = [];
  const rgb = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 7) continue;
    const [bx, by, bz] = applySceneRotation(
      parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])
    );
    xyz.push(bx, by, bz);
    rgb.push(parseInt(parts[4]) / 255, parseInt(parts[5]) / 255, parseInt(parts[6]) / 255);
  }
  return { xyz, rgb, count: xyz.length / 3 };
}

// Parse images.txt: each image is 2 lines (pose + 2D points). Returns {pose, name}[].
// COLMAP stores cam_from_world. We invert to world-from-cam for rendering.
function parseImages(text) {
  const images = [];
  // Keep blank lines: a camera with zero 2D points (posed-only annex cameras) has an EMPTY points
  // line, and dropping it shifts the pose/points pairing so every second such camera is eaten.
  // Only strip the comment header, then walk pose lines directly (a pose line has >= 10 fields
  // starting with an integer id; points lines are coordinate triplets or blank).
  const lines = text.split("\n").filter(l => !l.trim().startsWith("#"));
  for (let i = 0; i < lines.length; i += 2) {
    if (!lines[i].trim()) { i -= 1; continue; }  // stray blank in pose slot: resync
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length < 9) continue;
    const qw = parseFloat(parts[1]), qx = parseFloat(parts[2]),
          qy = parseFloat(parts[3]), qz = parseFloat(parts[4]);
    const tx = parseFloat(parts[5]), ty = parseFloat(parts[6]), tz = parseFloat(parts[7]);
    const name = parts.slice(9).join(" ");
    // World-from-cam: world_pos = -R^T @ t, where R is from quaternion.
    // R from (qw, qx, qy, qz):
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const R = [
      [1 - 2*(yy + zz),   2*(xy - wz),     2*(xz + wy)],
      [2*(xy + wz),       1 - 2*(xx + zz), 2*(yz - wx)],
      [2*(xz - wy),       2*(yz + wx),     1 - 2*(xx + yy)],
    ];
    // R^T @ t = column-by-column dot product (camera center in DATA coords)
    const cxC = -(R[0][0]*tx + R[1][0]*ty + R[2][0]*tz);
    const cyC = -(R[0][1]*tx + R[1][1]*ty + R[2][1]*tz);
    const czC = -(R[0][2]*tx + R[1][2]*ty + R[2][2]*tz);
    // Apply scene rotation so center is in viewer (Y-up aligned) coords.
    // Keep the camera rotation matrix R in DATA frame; renderFrustums will
    // rotate corner offsets through R then through sceneRotation.
    const [cx, cy, cz] = applySceneRotation(cxC, cyC, czC);
    // Tier flag: does this camera have any REAL 2D observation (POINT3D_ID != -1)? Posed-only
    // annex cameras have a blank or placeholder-only points line. Used to color flagged
    // cameras red in mixed models; models where no camera has observations (stripped viz
    // copies) are treated as untiered.
    let hasObs = false;
    const ptsLine = (lines[i + 1] || "").trim();
    if (ptsLine) {
      const f = ptsLine.split(/\s+/);
      for (let k = 2; k < f.length; k += 3) {
        if (f[k] !== "-1") { hasObs = true; break; }
      }
    }
    images.push({ position: [cx, cy, cz], R: R, name: name, dataPos: [cxC, cyC, czC], hasObs: hasObs });
  }
  return images;
}

// ── Scene rendering ─────────────────────────────────────────────────────────
function disposePoints() {
  if (pointsMesh) { pointsMesh.dispose(); pointsMesh = null; }
}
function disposeFrustums() {
  if (frustumLines) { frustumLines.dispose(); frustumLines = null; }
  if (frustumLinesAnnex) { frustumLinesAnnex.dispose(); frustumLinesAnnex = null; }
}

function renderStagePoints({ xyz, rgb, count }) {
  disposePoints();
  if (count === 0) return;

  const positions = new Float32Array(xyz);
  const colors = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    colors[i * 4 + 0] = rgb[i * 3 + 0];
    colors[i * 4 + 1] = rgb[i * 3 + 1];
    colors[i * 4 + 2] = rgb[i * 3 + 2];
    colors[i * 4 + 3] = 1.0;
  }
  const mesh = new BABYLON.Mesh("pts", scene);
  const vd = new BABYLON.VertexData();
  vd.positions = positions;
  vd.colors = colors;
  vd.applyToMesh(mesh, true);  // updatable=true so we can morph for LERP
  const mat = new BABYLON.StandardMaterial("pts_mat", scene);
  mat.disableLighting = true;
  mat.emissiveColor = new BABYLON.Color3(1, 1, 1);
  mat.pointsCloud = true;
  mat.pointSize = pointSize;
  mesh.material = mat;
  pointsMesh = mesh;
}

// Render dark wireframe frustums per camera. Snavely-2010 look: thin dark lines,
// uniform size, no shading. Frustum = 4 apex→corner lines + 4 corner-quad lines.
// Build the array of line segments for all frustums. Pure compute, no Babylon side effects.
function buildFrustumLines(images, viewRange) {
  // Frustum size = view_range * frustumScale (slider-controlled; default halved vs the old
  // fixed 0.03 — big frustum crowds obscured the structure on dense annex/GLOMAP models).
  const s = Math.max(0.001, viewRange * frustumScale);
  const half = s * 0.5;
  const localCorners = [
    [-half, -half, s],
    [ half, -half, s],
    [ half,  half, s],
    [-half,  half, s],
  ];
  const lines = [];
  for (const img of images) {
    const apex = new BABYLON.Vector3(img.position[0], img.position[1], img.position[2]);
    const worldCorners = localCorners.map(([lx, ly, lz]) => {
      // R^T @ local in DATA frame, then apply sceneRotation.
      const wxC = img.R[0][0]*lx + img.R[1][0]*ly + img.R[2][0]*lz;
      const wyC = img.R[0][1]*lx + img.R[1][1]*ly + img.R[2][1]*lz;
      const wzC = img.R[0][2]*lx + img.R[1][2]*ly + img.R[2][2]*lz;
      const [bx, by, bz] = applySceneRotation(wxC, wyC, wzC);
      return new BABYLON.Vector3(apex.x + bx, apex.y + by, apex.z + bz);
    });
    for (const c of worldCorners) lines.push([apex, c]);
    for (let i = 0; i < 4; i++) lines.push([worldCorners[i], worldCorners[(i + 1) % 4]]);
  }
  return lines;
}

function renderFrustums(images, viewRange) {
  disposeFrustums();
  if (!images.length) return;
  // Uniform red frustums (paper-wide convention: red = camera). The hasObs tier flag is still
  // parsed and available if two-tone rendering is ever wanted again.
  frustumLines = BABYLON.MeshBuilder.CreateLineSystem("frustums", {
    lines: buildFrustumLines(images, viewRange),
    updatable: true,
  }, scene);
  frustumLines.color = new BABYLON.Color3(0.85, 0.15, 0.15);  // muted red, like matplotlib
  frustumLines.alpha = 0.85;
}

// Update existing frustum mesh in place (no dispose/recreate). Babylon updates
// vertex positions of the same buffer when `instance:` is passed.
function updateFrustumsInPlace(images, viewRange) {
  if (!frustumLines || frustumLinesAnnex) {
    // Tiered models keep two line systems — rebuild both rather than instance-update one.
    renderFrustums(images, viewRange);
    return;
  }
  const lines = buildFrustumLines(images, viewRange);
  frustumLines = BABYLON.MeshBuilder.CreateLineSystem("frustums", {
    lines: lines,
    instance: frustumLines,
  }, scene);
}

// Smooth LERP between two stages' camera frusta. Component-wise lerp for both
// position + rotation matrix (not a true slerp, but visually fine for small
// per-stage refinements). Falls back to discrete swap if camera count differs.
function lerpFrustums(prevImgs, nextImgs, viewRange, durationMs) {
  if (frustumLerpRAF) cancelAnimationFrame(frustumLerpRAF);
  if (frustumLinesAnnex) {  // tiered models: discrete swap (lerp assumes one line system)
    renderFrustums(nextImgs, viewRange);
    return;
  }
  if (!prevImgs || !frustumLines || prevImgs.length !== nextImgs.length) {
    renderFrustums(nextImgs, viewRange);
    return;
  }
  const t0 = performance.now();
  const tick = () => {
    const elapsed = performance.now() - t0;
    const u = Math.min(1, elapsed / durationMs);
    const e = u * u * (3 - 2 * u);  // smoothstep
    const oneMinusE = 1 - e;
    // Build interpolated images (positions + rotations).
    const interp = nextImgs.map((nx, i) => {
      const pv = prevImgs[i];
      return {
        position: [
          pv.position[0] * oneMinusE + nx.position[0] * e,
          pv.position[1] * oneMinusE + nx.position[1] * e,
          pv.position[2] * oneMinusE + nx.position[2] * e,
        ],
        R: [
          [pv.R[0][0]*oneMinusE+nx.R[0][0]*e, pv.R[0][1]*oneMinusE+nx.R[0][1]*e, pv.R[0][2]*oneMinusE+nx.R[0][2]*e],
          [pv.R[1][0]*oneMinusE+nx.R[1][0]*e, pv.R[1][1]*oneMinusE+nx.R[1][1]*e, pv.R[1][2]*oneMinusE+nx.R[1][2]*e],
          [pv.R[2][0]*oneMinusE+nx.R[2][0]*e, pv.R[2][1]*oneMinusE+nx.R[2][1]*e, pv.R[2][2]*oneMinusE+nx.R[2][2]*e],
        ],
      };
    });
    updateFrustumsInPlace(interp, viewRange);
    if (u < 1) {
      frustumLerpRAF = requestAnimationFrame(tick);
    } else {
      frustumLerpRAF = null;
    }
  };
  frustumLerpRAF = requestAnimationFrame(tick);
}

// Paintball animation — for retri/densify stages. Each new point fires from a
// random camera frustum out into the scene, with per-point delay so they pepper
// in over time (not all at once). Phase 1: previous mesh fades out; Phase 2:
// fresh mesh built at camera positions, points fly to their final positions.
// Looks like a paintball gun spraying paint into the scene.
function paintballStagePoints(prevImages, next, durationMs) {
  if (lerpAnimRAF) cancelAnimationFrame(lerpAnimRAF);
  if (!next.count || !prevImages || prevImages.length === 0) {
    renderStagePoints(next);
    return;
  }
  // Pick a random source camera per point.
  const camPositions = prevImages.map(img => img.position);
  const nCams = camPositions.length;
  const startPos = new Float32Array(next.count * 3);
  for (let i = 0; i < next.count; i++) {
    const c = camPositions[Math.floor(Math.random() * nCams)];
    startPos[i * 3]     = c[0];
    startPos[i * 3 + 1] = c[1];
    startPos[i * 3 + 2] = c[2];
  }
  // Per-point fire delay (0-50% of duration) → staggered, paintball spray feel.
  const delays = new Float32Array(next.count);
  for (let i = 0; i < next.count; i++) delays[i] = Math.random() * 0.5;

  // Build the new mesh AT the start (camera) positions, with final colors.
  renderStagePoints({ xyz: Array.from(startPos), rgb: next.rgb, count: next.count });
  const endPos = new Float32Array(next.xyz);

  // Per-point alpha: hidden until the delay fires, then 1.0 (so points pop in
  // rather than streaking from the camera since frame 0). Babylon points-cloud
  // material doesn't support per-vertex alpha cheaply, so we just animate
  // position from camera → final and let perspective handle the "shoot" motion.
  const t0 = performance.now();
  const tick = () => {
    if (!pointsMesh) { lerpAnimRAF = null; return; }
    const elapsed = performance.now() - t0;
    const u = Math.min(1, elapsed / durationMs);
    const pos = pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    for (let i = 0; i < next.count; i++) {
      // Each point's local progress: starts at delay, finishes at 1.
      const localU = Math.max(0, Math.min(1, (u - delays[i]) / (1 - delays[i] + 1e-9)));
      // Exponential ease-out — fast initial flight, slows on approach (paintball arc).
      const e = 1 - Math.pow(1 - localU, 3);
      const ix = i * 3;
      pos[ix]     = startPos[ix]     * (1 - e) + endPos[ix]     * e;
      pos[ix + 1] = startPos[ix + 1] * (1 - e) + endPos[ix + 1] * e;
      pos[ix + 2] = startPos[ix + 2] * (1 - e) + endPos[ix + 2] * e;
    }
    pointsMesh.updateVerticesData(BABYLON.VertexBuffer.PositionKind, pos);
    if (u < 1) {
      lerpAnimRAF = requestAnimationFrame(tick);
    } else {
      lerpAnimRAF = null;
    }
  };
  lerpAnimRAF = requestAnimationFrame(tick);
}

// Smooth LERP between two point clouds. If counts differ, just discrete-swap
// (can't morph different vertex counts cleanly without correspondence).
function lerpStagePoints(prev, next, durationMs) {
  if (lerpAnimRAF) cancelAnimationFrame(lerpAnimRAF);
  if (!prev || !pointsMesh || prev.count !== next.count) {
    renderStagePoints(next);
    return;
  }
  const positions = pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
  const start = new Float32Array(positions);  // current shown positions
  const end = new Float32Array(next.xyz);
  const t0 = performance.now();
  const tick = () => {
    const elapsed = performance.now() - t0;
    const u = Math.min(1, elapsed / durationMs);
    const e = u * u * (3 - 2 * u);  // smoothstep ease
    for (let i = 0; i < positions.length; i++) {
      positions[i] = start[i] * (1 - e) + end[i] * e;
    }
    pointsMesh.updateVerticesData(BABYLON.VertexBuffer.PositionKind, positions);
    if (u < 1) {
      lerpAnimRAF = requestAnimationFrame(tick);
    } else {
      lerpAnimRAF = null;
      // Swap colors at the end (final stage's RGB).
      const colors = new Float32Array(next.count * 4);
      for (let i = 0; i < next.count; i++) {
        colors[i * 4 + 0] = next.rgb[i * 3 + 0];
        colors[i * 4 + 1] = next.rgb[i * 3 + 1];
        colors[i * 4 + 2] = next.rgb[i * 3 + 2];
        colors[i * 4 + 3] = 1.0;
      }
      pointsMesh.updateVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
    }
  };
  lerpAnimRAF = requestAnimationFrame(tick);
}

// Apply view setup computed via compute_up_alignment + compute_view_setup.
// Called once per RUN (not per stage) — sceneRotation, sceneViewCenter and
// sceneViewRange are scene-wide invariants.
function applyViewSetup(transformedPoints, transformedCamPositions) {
  const setup = computeViewSetup(transformedPoints);
  if (!setup) return;
  sceneViewCenter = setup.center;
  sceneViewRange = setup.range;
  const angles = computeAutoCameraAngles(transformedPoints, transformedCamPositions);
  camera.target = new BABYLON.Vector3(setup.center[0], setup.center[1], setup.center[2]);
  // Radius ≈ 2.5 × view_range so the 90th-pct sphere sits comfortably in frame.
  camera.radius = setup.range * 2.5;
  camera.alpha = angles.alpha;
  camera.beta = angles.beta;
  // Snapshot the view for "R" reset key.
  savedView = { center: setup.center, alpha: angles.alpha, beta: angles.beta, radius: setup.range * 2.5 };
}

// ── Stage loading ───────────────────────────────────────────────────────────
async function loadStage(idx, { animate = true } = {}) {
  if (!currentRun) return;
  const stage = currentRun.manifest.stages[idx];
  if (!stage) return;
  const stageUrl = stage.subdir
    ? `${currentRun.data_url_prefix}/${encodeURIComponent(stage.subdir)}`
    : currentRun.data_url_prefix;  // static reconstruction: files live at the run-dir root
  const [pointsText, imagesText] = await Promise.all([
    fetchText(`${stageUrl}/points3D.txt`),
    fetchText(`${stageUrl}/images.txt`),
  ]);
  const points = parsePoints3D(pointsText);
  const images = parseImages(imagesText);

  // Paintball effect ONLY for retri — that's where the structure dramatically
  // rebuilds (multi-view re-triangulation creates a fresh point set from the
  // correspondence graph). Densify is a quieter additive step where the existing
  // structure stays stable and we just lerp in 2-view supplements.
  const isPaintball = stage.stage_name === "retri";
  if (idx === 0 || !animate || !pointsMesh) {
    renderStagePoints(points);
  } else if (isPaintball && prevStageImages) {
    paintballStagePoints(prevStageImages, points, 1800);
  } else {
    lerpStagePoints(prevStageData, points, LERP_MS);
  }
  // Frustums: smooth LERP same as points when camera count matches. Otherwise
  // (rare; camera count is stable across stages in our pipeline) discrete swap.
  if (idx === 0 || !animate || !frustumLines) {
    renderFrustums(images, sceneViewRange ?? 1.0);
  } else {
    lerpFrustums(prevStageImages, images, sceneViewRange ?? 1.0, LERP_MS);
  }
  prevStageData = points;
  prevStageImages = images;

  // Update HUD
  document.getElementById("stageLabel").textContent =
    `${stage.stage_name} (${idx + 1}/${currentRun.manifest.stages.length})`;
  document.getElementById("numCameras").textContent = images.length.toLocaleString();
  document.getElementById("numPoints").textContent = points.count.toLocaleString();
  document.getElementById("stageIdx").textContent =
    `${idx + 1} / ${currentRun.manifest.stages.length}`;
  document.getElementById("timeline").value = idx;
  currentStageIdx = idx;
}

// Trim a manifest to the camera-ready set: first 10 GP iters + final, first 5
// of each per-iter BA group + final, plus all singleton stages (ba_input,
// outer_ba_iter_*, retri, retri_final_ba, per_obs_filter, min_tri_filter, densify).
function trimManifest(manifest, gpHead = 10, baHead = 5) {
  // Group stages by their iter-prefix (everything before "_iter_NNN").
  const groups = {};   // prefix → ordered array of {idx, stage}
  manifest.stages.forEach((s, idx) => {
    const name = s.stage_name;
    let prefix = null;
    if (name.startsWith("gp_iter_")) prefix = "gp";
    else if (name.includes("_iter_")) prefix = name.replace(/_iter_\d+$/, "");
    if (prefix) {
      (groups[prefix] = groups[prefix] || []).push({ idx, stage: s });
    }
  });
  const keep = new Set();
  for (const [prefix, entries] of Object.entries(groups)) {
    const headN = prefix === "gp" ? gpHead : baHead;
    for (let i = 0; i < Math.min(headN, entries.length); i++) keep.add(entries[i].idx);
    keep.add(entries[entries.length - 1].idx);  // always include last
  }
  const trimmed = manifest.stages.filter((s, idx) => {
    const name = s.stage_name;
    const isIter = name.startsWith("gp_iter_") || name.includes("_iter_");
    return !isIter || keep.has(idx);
  });
  return { ...manifest, stages: trimmed };
}

async function selectRun(runId) {
  const all = await (await fetch("/api/runs")).json();
  const item = all.items.find(r => r.id === runId);
  if (!item) return;
  let manifest = await (await fetch(item.manifest_url)).json();
  const mode = document.getElementById("modeSelect").value;
  if (mode === "trim") {
    const before = manifest.stages.length;
    manifest = trimManifest(manifest);
    console.log(`[trim] ${before} → ${manifest.stages.length} stages`);
  }
  currentRun = { ...item, manifest };
  document.getElementById("runLabel").textContent = item.label;
  const tl = document.getElementById("timeline");
  tl.max = String(Math.max(0, manifest.stages.length - 1));
  tl.value = "0";

  // Static reconstruction (single stage) → hide the trace-only controls (mode / play / timeline / stage).
  const isStatic = manifest.stages.length <= 1;
  for (const el of document.querySelectorAll('[data-role="trace-only"]')) {
    el.style.display = isStatic ? "none" : "";
  }

  // ── Compute scene-wide up-alignment + view setup from a REPRESENTATIVE stage.
  // First GP stages are noisy (random init) — pick the LAST stage (most converged
  // structure) so the up-alignment uses meaningful camera rotations + landmarks.
  // Reset rotation first so this stage's parse is in raw data coords.
  sceneRotation = null;
  sceneViewCenter = null;
  sceneViewRange = null;
  prevStageData = null;
  prevStageImages = null;
  disposePoints(); disposeFrustums();
  const lastStage = manifest.stages[manifest.stages.length - 1];
  const lastUrl = lastStage.subdir
    ? `${item.data_url_prefix}/${encodeURIComponent(lastStage.subdir)}`
    : item.data_url_prefix;
  const [refPointsText, refImagesText] = await Promise.all([
    fetchText(`${lastUrl}/points3D.txt`),
    fetchText(`${lastUrl}/images.txt`),
  ]);
  const refPoints = parsePoints3D(refPointsText);   // identity rotation right now
  const refImages = parseImages(refImagesText);
  const camPositions = refImages.map(img => img.dataPos);
  const camRotations = refImages.map(img => img.R);
  const landmarkArr = [];
  for (let i = 0; i < refPoints.count; i++) {
    landmarkArr.push([refPoints.xyz[i*3], refPoints.xyz[i*3+1], refPoints.xyz[i*3+2]]);
  }
  sceneRotation = computeUpAlignment(camPositions, camRotations, landmarkArr);
  // Apply the rotation to landmarks AND camera positions for view setup.
  const transformedPoints = landmarkArr.map(([x, y, z]) => applySceneRotation(x, y, z));
  const transformedCamPositions = camPositions.map(([x, y, z]) => applySceneRotation(x, y, z));
  applyViewSetup(transformedPoints, transformedCamPositions);

  // Now load stage 0 fresh — parsers will apply sceneRotation automatically.
  await loadStage(0);
}

// ── UI wiring ───────────────────────────────────────────────────────────────
async function init() {
  const data = await (await fetch("/api/runs")).json();
  const sel = document.getElementById("runSelect");
  for (const run of data.items) {
    const opt = document.createElement("option");
    opt.value = run.id;
    opt.textContent = `${run.label} (${run.num_stages} stages)`;
    sel.appendChild(opt);
  }
  if (data.items.length > 0) {
    sel.value = data.items[0].id;
    await selectRun(data.items[0].id);
  } else {
    document.getElementById("runLabel").textContent = "(no runs found)";
  }
  sel.addEventListener("change", () => selectRun(sel.value));

  // Trim-mode toggle: re-fetch + re-filter the manifest, reset to stage 0.
  document.getElementById("modeSelect").addEventListener("change", () =>
    selectRun(sel.value));

  document.getElementById("timeline").addEventListener("input", e =>
    loadStage(parseInt(e.target.value)));

  // Point-size slider — live update without rebuilding the mesh. Defensive
  // null-check for stale-cached HTML / pre-refresh state.
  const sizeSlider = document.getElementById("pointSize");
  if (sizeSlider) {
    pointSize = parseFloat(sizeSlider.value) || pointSize;
    sizeSlider.addEventListener("input", e => {
      pointSize = parseFloat(e.target.value);
      if (pointsMesh && pointsMesh.material) {
        pointsMesh.material.pointSize = pointSize;
      }
    });
  }

  // Frustum-size slider — rebuilds the line system in place from the current stage's images.
  const frustumSlider = document.getElementById("frustumSize");
  if (frustumSlider) {
    frustumScale = parseFloat(frustumSlider.value) || frustumScale;
    frustumSlider.addEventListener("input", e => {
      frustumScale = parseFloat(e.target.value);
      if (prevStageImages) {
        updateFrustumsInPlace(prevStageImages, sceneViewRange ?? 1.0);
      }
    });
  }

  const playBtn = document.getElementById("playBtn");
  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "❚❚" : "▶";
    if (isPlaying) {
      playTimer = setInterval(() => {
        const next = (currentStageIdx + 1) % currentRun.manifest.stages.length;
        loadStage(next);  // smooth LERP runs in the background while we wait
      }, LERP_MS + 200);  // step a hair after the LERP finishes
    } else {
      clearInterval(playTimer);
    }
  });
}

init().catch(err => {
  console.error(err);
  document.getElementById("runLabel").textContent = "ERROR — check server logs";
});
