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

test('source XY plane is horizontal and positive source Z points up', () => {
  const points = proto._parsePoints('1 2 3 0 255 0 0\n2 -4 5 0 0 255 0\n3 2 3 7 0 0 255\n');
  close(points[0].y, 0);
  close(points[1].y, 0);
  close(points[2].y, 7);
  close(points[0].x, 2);
  close(points[0].z, 3);
});

test('Z-up conversion preserves photo projection and frustum corners for a rolled camera', () => {
  const v = fixture();
  // Independent COLMAP world->camera rotation from a unit quaternion.
  const [w, x, y, z] = [0.8, 0.2, -0.3, 0.4].map(a => a / Math.sqrt(0.93));
  const R = [
    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
  ];
  const C = [3, 4, 5];
  const t = R.map(row => -row.reduce((sum, value, i) => sum + value*C[i], 0));
  const cameraPointToWorld = p => C.map((c, i) => c + R.reduce((sum, row, j) => sum + row[i]*p[j], 0));
  v.cameras = v._parseCams(`7 ${w} ${x} ${y} ${z} ${t.join(' ')} 1 rolled.jpg\n\n`);
  const cam = v.cameras[0];
  cam.intrinsics = { width: 800, height: 600, fx: 400, fy: 400, cx: 400, cy: 300 };
  v._autoFrame();
  v.camera.getViewMatrix(true);
  const world = cameraPointToWorld([1, -0.5, 4]);
  const point = v._parsePoints(`1 ${world.join(' ')} 255 255 255`)[0];
  const projected = BABYLON.Vector3.Project(new BABYLON.Vector3(point.x, point.y, point.z),
    BABYLON.Matrix.Identity(), v.camera.getViewMatrix().multiply(v.camera.getProjectionMatrix(true)),
    new BABYLON.Viewport(0, 0, 800, 600));
  close(projected.x, 500, 1e-3);
  close(projected.y, 250, 1e-3);
  const corners = v._imageCorners(cam, 2);
  const expectedCameraCorners = [[-2,-1.5,2], [2,-1.5,2], [2,1.5,2], [-2,1.5,2]];
  corners.forEach((corner, i) => {
    const actual = BABYLON.Vector3.TransformCoordinates(corner, cam.R).add(cam.center);
    const expected = cameraPointToWorld(expectedCameraCorners[i]);
    close(actual.x, expected[0]); close(actual.y, expected[2]); close(actual.z, expected[1]);
  });
  v.engine.dispose();
});

test('splat mesh transforms positions and anisotropic axes into the same Z-up frame', async () => {
  const v = fixture();
  // Use a real Babylon mesh for transforms; GPU splat uploads need a browser.
  class SplatMesh extends BABYLON.Mesh {
    constructor(name, url, scene) { super(name, scene); }
    async updateDataAsync() {}
  }
  sandbox.BABYLON = { ...BABYLON, GaussianSplattingMesh: SplatMesh };
  try {
    await v._createSplatsMesh({ buffer: new ArrayBuffer(32) });
    const matrix = v.splatsMesh.getWorldMatrix();
    const p = BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(2, 3, 7), matrix);
    close(p.x, 2); close(p.y, 7); close(p.z, 3);
    // Unequal principal axes must rotate with the centers, not stay in the old frame.
    for (const [axis, expected] of [
      [[2,0,0], [2,0,0]], [[0,3,0], [0,0,3]], [[0,0,4], [0,4,0]],
    ]) {
      const actual = BABYLON.Vector3.TransformNormal(BABYLON.Vector3.FromArray(axis), matrix);
      actual.asArray().forEach((value, i) => close(value, expected[i]));
    }
  } finally {
    sandbox.BABYLON = BABYLON;
    v.engine.dispose();
  }
});

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
  close(v.camera.position.y, -1);
  close(v.camera.position.z, -2);
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
