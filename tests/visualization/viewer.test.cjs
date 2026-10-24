/** Geometry and resource-budget regression tests, using Babylon's real NullEngine. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const BABYLON = require('../../visualization/frontend/node_modules/babylonjs');
const sandbox = { BABYLON, console, performance, queueMicrotask, document: { hidden: false }, window: { addEventListener() {}, dispatchEvent() {} }, CustomEvent: class {}, URLSearchParams, AbortController };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(require.resolve('../../visualization/static/viewer.js'), 'utf8') + '\nthis.Viewer = ColmapViewer;', sandbox);
const proto = sandbox.Viewer.prototype;

function fixture() {
  const engine = new BABYLON.NullEngine({ renderWidth: 800, renderHeight: 600 });
  const scene = new BABYLON.Scene(engine);
  const camera = new BABYLON.ArcRotateCamera('view', 0, 1, 5, BABYLON.Vector3.Zero(), scene);
  const v = Object.assign(Object.create(proto), {
    engine, scene, camera, frustumGroup: new BABYLON.TransformNode('frusta', scene),
    cameras: [], camIndex: 0, cameraLineMeshes: [], frustumLines: [], photoItems: [],
    photoTextures: new Map(), photoTextureLimit: 48, photoVisibleLimit: 32, photoGeneration: 0,
    showCams: true, sceneCenter: new BABYLON.Vector3(0, 0, 10),
    _renderStats() {}, _emitPhotoState() {},
  });
  return v;
}

function close(a, b, tolerance=1e-5) { assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`); }

test('COLMAP parsing retains spaced names, skips empty/negative observations and resolves intrinsics', () => {
  const cameras = proto._parseCams('# header\n7 1 0 0 0 0 0 0 3 first photo.jpg\n\n8 1 0 0 0 0 0 0 4 second.jpg\n-2.0 3.0 -1\n');
  assert.equal(cameras.length, 2);
  assert.equal(cameras[0].name, 'first photo.jpg');
  assert.equal(cameras[1].cameraId, 4);
  const k = proto._parseIntrinsics('3 RADIAL 800 600 400 400 300 0 0\n4 PINHOLE 800 600 420 410 400 300\n');
  assert.equal(k.get(3).fy, 400);
  assert.equal(k.get(4).fy, 410);
});

test('first camera uses reconstructed position, forward direction, roll and field of view', () => {
  const v = fixture();
  const q = Math.sqrt(0.5);
  v.cameras = v._parseCams(`7 ${q} 0 ${q} 0 1 2 3 3 frame.jpg\n\n`);
  v.cameras[0].intrinsics = { width: 800, height: 600, fx: 400, fy: 400, cx: 400, cy: 300 };
  v._autoFrame();
  v.camera.getViewMatrix(true);
  close(v.camera.position.x, 3);
  close(v.camera.position.y, 2);
  close(v.camera.position.z, -1);
  const forward = v.camera.getForwardRay().direction;
  close(forward.x, -1);
  close(forward.y, 0);
  close(forward.z, 0);
  close(v.camera.fov, 2 * Math.atan(0.75));
  assert.equal(v.camIndex, 0);
  v.frameScene();
  v._updateCameraFov(); // Collapsing a sidebar must preserve the overview FOV.
  close(v.camera.fov, 0.8);
  v.engine.dispose();
});

test('all frustums share one wireframe mesh and photo corners respect calibration', () => {
  const v = fixture();
  v.cameras = Array.from({ length: 1000 }, (_, i) => ({ name: `${i}.jpg`, center: new BABYLON.Vector3(i * .01, 0, 5), R: BABYLON.Matrix.Identity() }));
  v._createFrusta(v.cameras);
  assert.equal(v.cameraLineMeshes.length, 1);
  assert.equal(v.scene.meshes.length, 1);
  assert.equal(v.cameraLineMeshes[0].getTotalVertices(), 16000);
  const corners = v._imageCorners({ intrinsics: { width: 800, height: 600, fx: 400, fy: 400, cx: 400, cy: 300 } }, 2);
  close(corners[0].x, -2); close(corners[0].y, 1.5); close(corners[2].x, 2); close(corners[2].y, -1.5);
  v.engine.dispose();
});

test('texture work has bounded concurrency and releases resources on scene clear', async () => {
  const v = fixture();
  v.cameras = Array.from({ length: 200 }, (_, i) => ({ center: new BABYLON.Vector3(i * .001, 0, 5), R: BABYLON.Matrix.Identity() }));
  v.photoItems = v.cameras.map((_, index) => ({ thumbnail: `photo-${index}` }));
  v.frustumDepth = .1;
  v.camera.setPosition(new BABYLON.Vector3(0, 0, 0));
  v.camera.setTarget(new BABYLON.Vector3(0, 0, 5));
  v.camera.getViewMatrix(true); v.camera.getProjectionMatrix(true);
  const requested = [];
  class FakeTexture { static TRILINEAR_SAMPLINGMODE = 3; constructor(url, scene, noMip, flip, sampling, loaded) { requested.push(this); this.loaded = loaded; } dispose() { this.disposed = true; } }
  sandbox.BABYLON = { ...BABYLON, Texture: FakeTexture };
  v._updatePhotoTextures();
  assert.equal(requested.length, 4);
  v._updatePhotoTextures();
  assert.equal(requested.length, 4);
  // Finish each request; the scheduler stops at the visible texture budget.
  v._createPhotoPlane = () => {};
  for (let i = 0; i < requested.length; i++) { requested[i].loaded(); await Promise.resolve(); }
  assert.equal(requested.length, 32);
  assert.equal(v.photoTextures.size, 32);
  v._applyHudMode = () => {};
  v._clearSplats = () => {};
  await v._clear();
  assert.equal(v.photoTextures.size, 0);
  assert.ok(requested.every(texture => texture.disposed));
  requested[0].loaded(); await Promise.resolve(); // Late callbacks must not repopulate the cleared scene.
  assert.equal(v.photoTextures.size, 0);
  sandbox.BABYLON = BABYLON;
  v.engine.dispose();
});

test('photo material uses the texture without an additive white wash and disposes cleanly', () => {
  const v = fixture();
  v.cameras = [{ center: new BABYLON.Vector3(1, 2, 3), R: BABYLON.Matrix.Identity() }];
  v.frustumDepth = .2;
  const entry = { texture: new BABYLON.Texture(null, v.scene) };
  v._createPhotoPlane(0, entry);
  assert.equal(entry.material.emissiveTexture, entry.texture);
  assert.equal(entry.material.emissiveColor.r + entry.material.emissiveColor.g + entry.material.emissiveColor.b, 0);
  assert.equal(entry.mesh.metadata.cameraIndex, 0);
  assert.equal(entry.mesh.getTotalVertices(), 4);
  v._disposePhotoTexture(entry);
  assert.equal(v.scene.meshes.length, 0);
  assert.equal(v.scene.textures.length, 0);
  v.engine.dispose();
});
