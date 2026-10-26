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

test('live splat refresh retains the old mesh and viewpoint across slow downloads and failures', async () => {
  const v = fixture();
  class SplatMesh extends BABYLON.Mesh {
    async updateDataAsync() {}
    static async ConvertPLYWithSHToSplatAsync(buffer) { return { buffer }; }
  }
  sandbox.BABYLON = { ...BABYLON, GaussianSplattingMesh: SplatMesh };
  const loading = [];
  Object.assign(v, {
    mode: 'splat', _splatStreamId: 'job-1',
    _applyHudMode() {}, _setIdleGroundVisible() {},
    _setLoadingState(state) { loading.push(state); },
    _clear() { assert.fail('A live refresh must not clear the visible scene'); },
  });
  const previous = new BABYLON.Mesh('previous', v.scene);
  v.splatsMesh = previous;
  const cameraPosition = v.camera.position.clone();
  const cameraTarget = v.camera.target.clone();
  let releaseDownload;
  v._fetchResource = () => new Promise(resolve => { releaseDownload = resolve; });
  try {
    const pending = v.loadSplatsFile({ splatsUrl: '/preview?v=2', streamId: 'job-1' });
    assert.equal(v.splatsMesh, previous);
    assert.equal(previous.isDisposed(), false);
    assert.equal(loading[0].compact, true);
    releaseDownload(new ArrayBuffer(32));
    assert.equal(await pending, true);
    assert.notEqual(v.splatsMesh, previous);
    assert.equal(previous.isDisposed(), true);
    assert.equal(v.splatsMesh.isEnabled(), true);
    assert.ok(v.camera.position.equals(cameraPosition));
    assert.ok(v.camera.target.equals(cameraTarget));
    assert.equal(v.isBusy(), false);
    const successfulMesh = v.splatsMesh;
    v._fetchResource = async () => { throw Error('Connection interrupted'); };
    assert.equal(await v.loadSplatsFile({ splatsUrl: '/preview?v=3', streamId: 'job-1' }), false);
    assert.equal(v.splatsMesh, successfulMesh);
    assert.equal(successfulMesh.isDisposed(), false);
    assert.equal(v.isBusy(), false);
    assert.equal(loading.at(-1).active, false);
  } finally {
    sandbox.BABYLON = BABYLON;
    v.engine.dispose();
  }
});

test('overview ignores isolated point and camera outliers without excluding the camera group', () => {
  const v = fixture();
  const points = Array.from({length: 100}, (_, i) => new BABYLON.Vector3(i / 99, (i % 10) / 9, 0));
  points.push(new BABYLON.Vector3(1e8, -1e8, 1e8));
  const cameras = Array.from({length: 30}, (_, i) => new BABYLON.Vector3(10 + i / 29, 2, -3));
  cameras.push(new BABYLON.Vector3(-1e9, 1e9, -1e9));
  v.overviewPointBounds = v._robustBounds(points);
  v.cameras = cameras.map(center => ({center}));
  v.frameScene();
  const target = v.camera.target;
  assert.ok(target.x > 5 && target.x < 6, 'Camera group must contribute to the combined bounds');
  assert.ok(v.camera.radius < 20, 'Distant outliers must not force an enormous overview distance');
  assert.ok(v._sceneExtent() < 2, 'Camera markers must not scale with rejected point outliers');
  assert.equal(points.length, 101, 'Framing must not remove reconstructed points');
  assert.equal(v.cameras.length, 31);
  v.engine.dispose();
});

test('robust bounds handle empty, nonfinite, tiny, and point-only scenes', () => {
  const v = fixture();
  assert.equal(v._robustBounds([]), null);
  assert.equal(v._robustBounds([{x: NaN, y: 0, z: 0}]), null);
  const points = [new BABYLON.Vector3(2, 3, 4), new BABYLON.Vector3(8, 9, 10)];
  const bounds = v._robustBounds([...points, {x: Infinity, y: 0, z: 0}]);
  assert.ok(bounds.min.equals(points[0]));
  assert.ok(bounds.max.equals(points[1]));
  v.overviewPointBounds = bounds;
  v.frameScene();
  assert.ok(v.camera.target.equals(new BABYLON.Vector3(5, 6, 7)));
  assert.ok(Number.isFinite(v.camera.radius) && v.camera.radius > 0);
  v.overviewPointBounds = v._robustBounds([points[0]]);
  v.frameScene();
  assert.ok(Number.isFinite(v.camera.radius) && v.camera.radius > 0);
  v.engine.dispose();
});

test('camera-plane fit levels a tilted camera rig and chooses the camera-up hemisphere', () => {
  const v = fixture();
  const tilt = BABYLON.Matrix.RotationYawPitchRoll(0.3, 0.6, -0.4);
  v.cameras = Array.from({length: 25}, (_, i) => ({
    center: BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(i % 5, 3, Math.floor(i / 5)), tilt),
    R: tilt,
  }));
  const rotation = v._cameraPlaneRotation();
  const transformed = v.cameras.map(camera => BABYLON.Vector3.TransformCoordinates(camera.center, rotation));
  const heights = transformed.map(point => point.y);
  assert.ok(Math.max(...heights) - Math.min(...heights) < 1e-5);
  const normal = BABYLON.Vector3.TransformNormal(BABYLON.Vector3.Up(), tilt);
  const up = BABYLON.Vector3.TransformNormal(normal, rotation);
  close(up.x, 0); close(up.y, 1); close(up.z, 0);
  close(BABYLON.Vector3.Distance(transformed[0], transformed[24]), BABYLON.Vector3.Distance(v.cameras[0].center, v.cameras[24].center));
  v.engine.dispose();
});

test('plane alignment rejects insufficient, coincident, collinear, and isotropic camera centers', () => {
  const v = fixture();
  const set = points => { v.cameras = points.map(center => ({center, R: BABYLON.Matrix.Identity()})); };
  set([BABYLON.Vector3.Zero(), BABYLON.Vector3.One()]);
  assert.throws(() => v._cameraPlaneRotation(), /three camera centers/);
  set([BABYLON.Vector3.One(), BABYLON.Vector3.One(), BABYLON.Vector3.One()]);
  assert.throws(() => v._cameraPlaneRotation(), /coincide/);
  set(Array.from({length: 5}, (_, i) => new BABYLON.Vector3(i, i * 2, i * 3)));
  assert.throws(() => v._cameraPlaneRotation(), /stable plane/);
  set([new BABYLON.Vector3(1,0,0),new BABYLON.Vector3(-1,0,0),new BABYLON.Vector3(0,1,0),new BABYLON.Vector3(0,-1,0),new BABYLON.Vector3(0,0,1),new BABYLON.Vector3(0,0,-1)]);
  assert.throws(() => v._cameraPlaneRotation(), /stable plane/);
  v.engine.dispose();
});

test('Z-up alignment rotates points and camera poses together and can restore the original frame', () => {
  const v = fixture();
  v.mode = 'scene';
  const tilt = BABYLON.Matrix.RotationYawPitchRoll(-0.3, -0.5, 0.7);
  v.cameras = Array.from({length: 9}, (_, i) => ({
    center: BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(i % 3, 2, Math.floor(i / 3)), tilt), R: tilt,
  }));
  const cachedCameras = v.cameras;
  const originalCenter = cachedCameras[0].center.clone();
  v.pointsMesh = BABYLON.MeshBuilder.CreateBox('cloud', {size: 2}, v.scene);
  const originalPositions = Array.from(v.pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind));
  const originalPoint = BABYLON.Vector3.FromArray(originalPositions);
  const localBefore = BABYLON.Vector3.TransformNormal(originalPoint.subtract(v.cameras[0].center), BABYLON.Matrix.Invert(v.cameras[0].R));
  v.alignZUp();
  assert.ok(v.alignmentRotation);
  assert.equal(v.cameraLineMeshes.length, 1);
  assert.ok(cachedCameras[0].center.equals(originalCenter), 'Cached source cameras must be unchanged');
  const pointAfter = BABYLON.Vector3.FromArray(v.pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind));
  const localAfter = BABYLON.Vector3.TransformNormal(pointAfter.subtract(v.cameras[0].center), BABYLON.Matrix.Invert(v.cameras[0].R));
  localBefore.asArray().forEach((value, i) => close(value, localAfter.asArray()[i]));
  v._gotoCam(0);
  const forward = v.camera.getForwardRay().direction;
  const expected = BABYLON.Vector3.TransformNormal(new BABYLON.Vector3(0,0,1), v.cameras[0].R).normalize();
  assert.ok(BABYLON.Vector3.Dot(forward, expected) > 0.99999);
  v.alignZUp();
  assert.equal(v.alignmentRotation, null);
  Array.from(v.pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind)).forEach((value, i) => close(value, originalPositions[i]));
  v.cameras.forEach((camera, i) => camera.center.asArray().forEach((value, j) => close(value, cachedCameras[i].center.asArray()[j])));
  assert.equal(v.cameraLineMeshes.length, 1);
  v.engine.dispose();
});

test('ground visibility does not constrain zoom and close inspection moves the near plane inward', () => {
  const v = fixture();
  v._setZoomScale(10);
  const lower = v.camera.lowerRadiusLimit;
  assert.ok(lower < 0.001);
  v.showGround = true;
  v._setIdleGroundVisible(true);
  assert.equal(v.camera.lowerRadiusLimit, lower);
  assert.equal(v.camera.upperRadiusLimit, null);
  v.camera.radius = 0.005;
  v.camera.getViewMatrix(true); // Babylon applies radius constraints here.
  close(v.camera.radius, 0.005);
  v._updateCameraClipping();
  assert.ok(v.camera.minZ < 0.00001);
  v.showGround = false;
  v._setIdleGroundVisible(true);
  assert.equal(v.camera.lowerRadiusLimit, lower);
  v._setZoomScale(0.001);
  assert.ok(v.camera.lowerRadiusLimit <= 1e-8);
  v.engine.dispose();
});

test('point reveal restores the original material without changing reconstruction vertices', async () => {
  const v = fixture();
  v.ptSize = 2;
  v.sceneCenter = BABYLON.Vector3.Zero();
  await v._createPCS([{x: 1, y: 2, z: 3, r: 1, g: 0, b: 0}, {x: -1, y: 0, z: 1, r: 0, g: 1, b: 0}]);
  const mesh = v.pointsMesh;
  const material = mesh.material;
  const positions = Array.from(mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind));
  const observerCount = v.scene.onBeforeRenderObservable.observers.length;
  let now = 0;
  const originalPerformance = sandbox.performance;
  sandbox.performance = {now: () => now};
  try {
    v._startPointReveal();
    assert.notEqual(mesh.material, material);
    v._pointReveal.material.isReady = () => true;
    v.scene.onBeforeRenderObservable.notifyObservers(v.scene);
    now = 1300;
    v.scene.onBeforeRenderObservable.notifyObservers(v.scene);
    assert.equal(v._pointReveal, null);
    assert.equal(mesh.material, material);
    assert.equal(mesh.alwaysSelectAsActiveMesh, false);
    assert.deepEqual(Array.from(mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind)), positions);
    // Babylon marks observers for deferred removal; they must no longer be invoked.
    assert.equal(v.scene.onBeforeRenderObservable.observers.filter(o => !o._willBeUnregistered).length, observerCount);
    v._startPointReveal();
    v._stopPointReveal();
    v._stopPointReveal();
    assert.equal(mesh.material, material);
    sandbox.window.matchMedia = () => ({matches: true});
    v._startPointReveal();
    assert.equal(v._pointReveal, null);
    assert.equal(mesh.material, material);
  } finally {
    delete sandbox.window.matchMedia;
    sandbox.performance = originalPerformance;
    v.engine.dispose();
  }
});

test('streamed text reports byte progress, preserves split UTF-8, and reuses cached content', async () => {
  const bytes = new TextEncoder().encode('café reconstruction');
  const reports = [];
  const v = Object.assign(Object.create(proto), {resourceCache: new Map(), resourceCacheLimit: 8});
  sandbox.TextDecoder = TextDecoder;
  sandbox.fetch = async () => new Response(new ReadableStream({start(controller) {
    controller.enqueue(bytes.slice(0, 4)); // split the two-byte é
    controller.enqueue(bytes.slice(4));
    controller.close();
  }}), {headers: {'content-length': String(bytes.length)}});
  try {
    const result = await v._fetchResource('/points', 'text', {onProgress: (...args) => reports.push(args)});
    assert.equal(result, 'café reconstruction');
    assert.deepEqual(reports[0], [0, bytes.length]);
    assert.deepEqual(reports[1], [4, bytes.length]);
    assert.deepEqual(reports.at(-1), [bytes.length, bytes.length, true]);
    sandbox.fetch = () => { throw Error('Cache should avoid another download'); };
    assert.equal(await v._fetchResource('/points', 'text'), result);
    for (const headers of [{}, {'content-encoding': 'gzip', 'content-length': '2'}]) {
      sandbox.fetch = async () => new Response(bytes, {headers});
      const unknown = [];
      await v._fetchResource('/unknown', 'text', {cacheResult: false, onProgress: (...args) => unknown.push(args)});
      assert.equal(unknown[0][1], null, 'Do not invent a percentage for missing or encoded content lengths');
      assert.deepEqual(unknown.at(-1), [bytes.length, bytes.length, true]);
    }
  } finally { delete sandbox.fetch; delete sandbox.TextDecoder; }
});

test('incremental point batches use one mesh and retain exact vertices after buffer growth', () => {
  const v = fixture();
  v.ptSize = 2;
  const stream = {points: [], count: 0, capacity: 0, userNavigated: false};
  v._pointStream = stream;
  const append = (start, length) => {
    const batch = Array.from({length}, (_, k) => ({x: start + k, y: (start + k) % 7, z: 1, r: 1, g: 0.5, b: 0}));
    stream.points.push(...batch);
    v._appendPointBatch(stream, batch);
  };
  append(0, 128);
  const mesh = v.pointsMesh;
  assert.equal(v.pointsCount, 128);
  assert.equal(mesh.subMeshes[0].indexCount, 128);
  append(128, 100);
  assert.equal(mesh.subMeshes[0].indexCount, 228);
  append(228, 5000); // force an amortized buffer resize
  assert.equal(v.pointsMesh, mesh);
  assert.equal(v.pointsCount, 5228);
  assert.equal(v._pointReveal.streaming, true);
  v._finishPointStream(stream);
  assert.equal(v._pointReveal.streaming, false);
  const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
  assert.equal(positions.length, 5228 * 3);
  for (let i = 0; i < 5228; i++) {
    close(positions[i * 3], i);
    close(positions[i * 3 + 1], i % 7);
    close(positions[i * 3 + 2], 1);
  }
  v._stopPointReveal();
  assert.equal(mesh.alwaysSelectAsActiveMesh, false);
  assert.equal(mesh.isVerticesDataPresent('arrival'), false);
  v.engine.dispose();
});

test('3D points appear before EOF and finish without waiting for camera metadata', async () => {
  const v = fixture();
  Object.assign(v, {ptSize: 2, resourceCache: new Map(), resourceCacheLimit: 8,
    parsedPointsCache: new Map(), parsedCamsCache: new Map(), parsedCacheLimit: 4,
    _yieldFrame: async () => {}, _loadPhotos: async () => { throw Error('Photos should not block structure'); }});
  const first = Array.from({length: 128}, (_, i) => `${i} ${i} 2 3 255 128 0 0`).join('\n') + '\n';
  const text = first + '128 128 2 3 255 128 0 0'; // final line deliberately has no newline
  let reader, finishCamera;
  const cameraWait = new Promise(resolve => { finishCamera = resolve; });
  sandbox.TextDecoder = TextDecoder;
  sandbox.fetch = async url => url === '/points'
    ? new Response(new ReadableStream({start(controller) { reader = controller; }}), {headers: {'content-length': String(text.length)}})
    : cameraWait;
  try {
    const loading = v.loadScene({pointsUrl:'/points', imagesUrl:'/cameras'});
    while (!reader) await new Promise(resolve => setImmediate(resolve));
    reader.enqueue(new TextEncoder().encode(first));
    while (v.pointsCount !== 128) await new Promise(resolve => setImmediate(resolve));
    assert(v.pointsMesh, 'Mesh must exist while download is still open');
    assert(v.isBusy());
    assert.equal(v.cameraViewMode, 'overview');
    reader.enqueue(new TextEncoder().encode(text.slice(first.length)));
    reader.close();
    assert.equal(await loading, true);
    assert.equal(v.pointsCount, 129);
    assert.equal(v.isBusy(), false, 'Camera download must not hold the structure busy');
    assert.equal(v.parsedPointsCache.get('/points').length, 129);
    assert.equal(v.resourceCache.has('text:/points'), false, 'Do not retain a redundant raw point file');
    // Clearing aborts the stale metadata callback as well as the point animation.
    await v._clear();
    finishCamera(new Response(''));
    await v._sceneMetadataPromise;
    assert.equal(v.pointsCount, 0);
    assert.equal(v._pointReveal, null);
  } finally {
    finishCamera(new Response(''));
    delete sandbox.fetch; delete sandbox.TextDecoder;
    v.engine.dispose();
  }
});
