// visualization/static/viewer.js

class ColmapViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.055, 0.063, 0.071, 1.0);

    this.camera = new BABYLON.ArcRotateCamera(
      "cam",
      -Math.PI / 2,
      Math.PI / 2.2,
      15,
      new BABYLON.Vector3(0, 0, 0),
      this.scene
    );
    this.scene.activeCamera = this.camera;
    this.camera.attachControl(canvas, true);
    this.camera.minZ = 0.01;    // Near clipping plane
    this.camera.maxZ = 10000;   // Far clipping plane

    // Source Z-up is mapped to Babylon's Y-up, left-handed world frame.
    this.camera.upVector = new BABYLON.Vector3(0, 1, 0);
    this.camera.setPosition(new BABYLON.Vector3(10, 8, -12));
    this.camera.setTarget(BABYLON.Vector3.Zero());
    this.camera.rebuildAnglesAndRadius?.();


    // Inputs & limits tuned to avoid “locked” feeling
    this.camera.angularSensibilityX = 1500;
    this.camera.angularSensibilityY = 1500;
    this.camera.panningSensibility = 1000;
    this.camera.wheelPrecision = 90;
    this.camera.wheelDeltaPercentage = 0.01;
    this.camera.pinchPrecision = 120;
    this.camera.useNaturalPinchZoom = true;
    this._setZoomScale(1);
    this.camera.lowerBetaLimit = 0.05;
    this.camera.upperBetaLimit = Math.PI - 0.05;

    this.light = new BABYLON.HemisphericLight("H", new BABYLON.Vector3(0, 1, 0), this.scene);
    this.light.intensity = 0.85;

    this.groundY = 0;
    this.showGround = false;
    this.groundAvailable = true;
    this.groundGrid = null;
    this.groundAxes = null;
    this.idleGround = this._createIdleGround();
    this._setIdleGroundVisible(true);

    this.pcs = null;
    this.pointsMesh = null;
    this.frustumGroup = new BABYLON.TransformNode("frusta", this.scene);
    this.cameraLineMeshes = [];
    this.cameras = [];
    this.camIndex = 0;
    this.photoItems = [];
    this.photoLoading = false;
    this.photoError = "";
    this.photoAbort = null;
    this.photoTextures = new Map();
    this.photoListeners = new Set();
    this.photoTextureLimit = 48;
    this.photoVisibleLimit = 32;
    this.photoGeneration = 0;
    this.frustumLines = [];

    this.splatsUrl = null;
    this.splatsMesh = null;
    this._splatsLoadingPromise = null;
    this._activeSplatsToken = null;
    this._splatsMeshToken = null;
    this.splatsPointCount = 0;
    this.mode = "idle";

    this.ptSize = 2;
    this.showCams = true;

    this.pointsCount = 0;
    this.cameraCount = 0;
    this.currentImageName = "";
    this.statsVisible = true;
    this.loading = { active: false, message: "", progress: 0, compact: false };
    this.stateListeners = new Set();
    this.resourceCache = new Map();
    this.resourceCacheLimit = 8;
    this.parsedPointsCache = new Map();
    this.parsedCamsCache = new Map();
    this.parsedCacheLimit = 4;
    this._busyState = null;
    this.overlayTheme = "dark";
    this.setOverlayTheme(this.overlayTheme);
    this._restorePreferences();

    this.scene.onBeforeRenderObservable.add(() => this._updateCameraClipping());
    this.camera.onAfterCheckInputsObservable.add(() => this._updateOrthographicBounds());
    this.engine.runRenderLoop(() => this.scene.render());
    this.onResize = () => this.engine.resize();
    window.addEventListener("resize", this.onResize);
    // Sidebar collapse changes the canvas size without a window resize.
    this.resizeObserver = new ResizeObserver(() => { this.engine.resize(); this._updateCameraFov(); });
    this.resizeObserver.observe(canvas);
    this.photoTimer = window.setInterval(() => this._updatePhotoTextures(), 400);
    this.scene.onPointerObservable.add((event) => {
      const index = event.pickInfo?.pickedMesh?.metadata?.cameraIndex;
      if (event.type === BABYLON.PointerEventTypes.POINTERTAP && Number.isInteger(index)) this.selectCamera(index);
    });
  }

  _createIdleGround() {
    const root = new BABYLON.TransformNode("idle-ground", this.scene);

    const gridLines = [];
    for (let value = -5; value <= 5; value += 1) {
      gridLines.push([
        new BABYLON.Vector3(value, 0.01, -5),
        new BABYLON.Vector3(value, 0.01, 5),
      ]);
      gridLines.push([
        new BABYLON.Vector3(-5, 0.01, value),
        new BABYLON.Vector3(5, 0.01, value),
      ]);
    }
    const grid = BABYLON.MeshBuilder.CreateLineSystem("idle-grid", { lines: gridLines }, this.scene);
    grid.parent = root;
    grid.color = new BABYLON.Color3(0.19, 0.205, 0.21);
    grid.alpha = 0.72;
    grid.isPickable = false;
    this.groundGrid = grid;

    const axes = BABYLON.MeshBuilder.CreateLineSystem(
      "idle-axes",
      {
        lines: [
          [new BABYLON.Vector3(-5, 0.015, 0), new BABYLON.Vector3(5, 0.015, 0)],
          [new BABYLON.Vector3(0, 0.015, -5), new BABYLON.Vector3(0, 0.015, 5)],
        ],
      },
      this.scene
    );
    axes.parent = root;
    axes.color = new BABYLON.Color3(0.36, 0.38, 0.38);
    axes.alpha = 0.9;
    axes.isPickable = false;
    this.groundAxes = axes;
    return root;
  }

  _setIdleGroundVisible(visible) {
    this.groundAvailable = !!visible;
    const effectiveVisibility = this.groundAvailable && this.showGround;
    this.idleGround?.setEnabled(effectiveVisibility);
  }

  _setZoomScale(size) {
    const scale = Number.isFinite(size) && size > 0 ? size : 1;
    this.camera.lowerRadiusLimit = Math.max(1e-8, scale * 1e-6);
    this.camera.upperRadiusLimit = null;
    this.camera.maxZ = Math.max(10000, scale * 100);
    this._updateCameraClipping();
  }

  _updateCameraClipping() {
    // Move the near plane inward during close inspection instead of clipping
    // geometry at the near distance chosen for the initial overview.
    this.camera.minZ = Math.max(this.camera.lowerRadiusLimit * 0.01, this.camera.radius * 0.001);
  }

  toggleOrthographicProjection() {
    const orthographic = this.camera.mode !== BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
    this.setOrthographicProjection(orthographic);
  }

  setOrthographicProjection(orthographic) {
    this.camera.mode = orthographic ? BABYLON.Camera.ORTHOGRAPHIC_CAMERA : BABYLON.Camera.PERSPECTIVE_CAMERA;
    this._updateOrthographicBounds();
    this._emitState();
  }

  setBackground(theme) {
    const colors = {
      dark: [0.055, 0.063, 0.071, 1], graphite: [0.10, 0.11, 0.12, 1],
      "light-gray": [0.85, 0.85, 0.83, 1], white: [0.98, 0.98, 0.96, 1],
    };
    if (!Object.hasOwn(colors, theme)) return;
    this.scene.clearColor = new BABYLON.Color4(...colors[theme]);
    this.setOverlayTheme(theme);
  }

  _restorePreferences() {
    try {
      const saved = JSON.parse(window.localStorage.getItem("gtsfm.studio.viewer.v1"));
      if (!saved || typeof saved !== "object") return;
      if (typeof saved.background === "string") this.setBackground(saved.background);
      if (typeof saved.orthographic === "boolean") this.setOrthographicProjection(saved.orthographic);
      if (typeof saved.showPlane === "boolean") this.setGroundVisible(saved.showPlane);
      if (typeof saved.planeHeight === "number" && Number.isFinite(saved.planeHeight)) this.setGroundY(saved.planeHeight);
    } catch { /* Invalid or unavailable browser storage leaves the defaults in place. */ }
  }

  savePreferences() {
    try {
      window.localStorage.setItem("gtsfm.studio.viewer.v1", JSON.stringify({
        background: this.overlayTheme,
        orthographic: this.camera.mode === BABYLON.Camera.ORTHOGRAPHIC_CAMERA,
        showPlane: this.showGround,
        planeHeight: this.groundY,
      }));
    } catch { /* Viewer controls still work when storage is unavailable. */ }
  }

  _updateOrthographicBounds() {
    if (this.camera.mode !== BABYLON.Camera.ORTHOGRAPHIC_CAMERA) return;
    // Match the perspective view's scale at the orbit target. Deriving the
    // bounds from radius also preserves wheel/pinch zoom and overview framing.
    const halfHeight = Math.max(1e-10, this.camera.radius * Math.tan(this.camera.fov / 2));
    const halfWidth = halfHeight * this.engine.getAspectRatio(this.camera);
    this.camera.orthoLeft = -halfWidth;
    this.camera.orthoRight = halfWidth;
    this.camera.orthoBottom = -halfHeight;
    this.camera.orthoTop = halfHeight;
  }

  setGroundVisible(value) {
    this.showGround = !!value;
    this._setIdleGroundVisible(this.groundAvailable);
    this._emitState();
    return this.showGround;
  }

  toggleGroundVisible() {
    return this.setGroundVisible(!this.showGround);
  }

  isGroundVisible() {
    return this.showGround;
  }

  setOverlayTheme(theme) {
    this.overlayTheme = theme;
    const lightBackground = theme === "light-gray" || theme === "white";
    const cameraColor = lightBackground
      ? new BABYLON.Color3(0.08, 0.09, 0.1)
      : new BABYLON.Color3(0.96, 0.97, 0.98);
    const gridColor = lightBackground
      ? new BABYLON.Color3(0.56, 0.57, 0.57)
      : new BABYLON.Color3(0.19, 0.205, 0.21);
    const axesColor = lightBackground
      ? new BABYLON.Color3(0.31, 0.32, 0.32)
      : new BABYLON.Color3(0.36, 0.38, 0.38);
    if (this.groundGrid) this.groundGrid.color = gridColor;
    if (this.groundAxes) this.groundAxes.color = axesColor;
    for (const mesh of this.cameraLineMeshes ?? []) mesh.color = cameraColor.clone();
    this._emitState();
  }

  setPointSize(px) {
    this.ptSize = Math.max(0.5, Math.min(5, px));
    if (this.pointsMesh?.material) this.pointsMesh.material.pointSize = this.ptSize;
    this._emitState();
  }

  setGroundY(value) {
    const next = Number(value);
    if (!Number.isFinite(next)) return this.groundY;
    this.groundY = Math.max(-5, Math.min(5, next));
    if (this.idleGround) this.idleGround.position.y = this.groundY;
    this._emitState();
    return this.groundY;
  }

  setShowCams(v) {
    this.showCams = v;
    this.frustumGroup.setEnabled(!!v);
    this._emitState();
    this._syncCameraVisuals();
  }

  _syncCameraVisuals() {
    const selected = this.cameras[this.camIndex];
    this.hiddenCamera = selected && BABYLON.Vector3.DistanceSquared(this.camera.position, selected.center) < this.frustumDepth ** 2 ? this.camIndex : -1;
    if (this.frustumLines.length && this.cameraLineMeshes[0]) {
      // One draw call for all wireframes. Degenerate the active camera's lines
      // so its own frustum never obscures the photo-aligned view.
      const lines = this.frustumLines.flatMap((lines, index) => index === this.hiddenCamera
        ? lines.map(([point]) => [point, point]) : lines);
      BABYLON.MeshBuilder.CreateLineSystem("camera-wireframes", { lines, instance: this.cameraLineMeshes[0] });
    }
    this._updatePhotoTextures();
  }

  onFirstFrame(listener) { this.scene.onAfterRenderObservable.addOnce(listener); }

  // Plain snapshots form the boundary with React; the renderer never mutates UI DOM.
  getState() {
    return {
      busy: this.isBusy(), mode: this.mode, background: this.overlayTheme,
      orthographic: this.camera.mode === BABYLON.Camera.ORTHOGRAPHIC_CAMERA,
      showGround: this.showGround, groundY: this.groundY, showCams: this.showCams,
      pointSize: this.ptSize, statsVisible: this.statsVisible,
      points: this.pointsCount, cameras: this.cameraCount, splats: this.splatsPointCount,
      imageName: this.currentImageName, aligned: !!this.alignmentRotation,
      alignmentMessage: this.alignmentMessage || "", loading: { ...this.loading },
    };
  }

  subscribe(listener) {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  _emitState() {
    if (!this.stateListeners?.size || this.stateUpdateQueued || this.disposed) return;
    this.stateUpdateQueued = true;
    queueMicrotask(() => {
      this.stateUpdateQueued = false;
      if (!this.disposed) for (const listener of this.stateListeners) listener(this.getState());
    });
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.stateListeners.clear();
    this.photoListeners.clear();
    this._busyState = null;
    void this._clear();
    window.removeEventListener("resize", this.onResize);
    this.resizeObserver.disconnect();
    window.clearInterval(this.photoTimer);
    this.resourceCache.clear();
    this.parsedPointsCache.clear();
    this.parsedCamsCache.clear();
    this.engine.stopRenderLoop();
    this.scene.dispose();
    this.engine.dispose();
  }

  isBusy() {
    return !!this._busyState;
  }

  _beginBusy(reason = null) {
    if (this._busyState) {
      console.warn("Overwriting existing busy state:", this._busyState.reason, "→", reason);
    }
    const token = Symbol("busy");
    this._busyState = { token, reason: reason ?? null };
    this._notifyBusyChange();
    return token;
  }

  _endBusy(token) {
    if (!this._busyState || this._busyState.token !== token) return;
    this._busyState = null;
    this._notifyBusyChange();
  }

  _notifyBusyChange() { this._emitState(); }

  setStatsVisible(value) {
    this.statsVisible = !!value;
    this._applyStatsVisibility();
  }

  toggleStatsVisible() {
    this.setStatsVisible(!this.statsVisible);
  }

  isStatsVisible() {
    return this.statsVisible;
  }

  _applyStatsVisibility() { this._emitState(); }
  _applyHudMode() { this._emitState(); }
  _renderStats() { this._emitState(); }

  _setLoadingState({ active = null, message = null, progress = null, compact = false } = {}) {
    this.loading = { ...this.loading };
    if (active !== null) { this.loading.active = active; this.loading.compact = compact; }
    if (message !== null) this.loading.message = message;
    if (progress !== null && Number.isFinite(progress)) this.loading.progress = Math.max(0, Math.min(1, progress));
    this._emitState();
  }

  _updateLoadingProgress(progress) {
    this._setLoadingState({ progress });
  }

  async _yieldFrame() {
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }

  async loadScene({ pointsUrl, imagesUrl, camerasUrl = null, scenePath = null, splatsUrl = null }) {
    if (this.disposed) return false;
    const busyToken = this._beginBusy("scene");
    await this._clear();
    if (this._busyState?.token !== busyToken) return false;
    const controller = new AbortController();
    this._sceneLoadAbort = controller;
    const { signal } = controller;
    const current = () => !signal.aborted && this._sceneLoadAbort === controller;
    this._setIdleGroundVisible(true);
    this.mode = "scene";
    this.splatsUrl = splatsUrl;
    this._applyHudMode();
    this._setLoadingState({ active: true, compact: true, message: "Loading 3D points…", progress: 0 });

    // Camera poses/intrinsics are metadata, not photographs. Neither gates the point stream.
    const cachedCameras = imagesUrl ? this.parsedCamsCache.get(imagesUrl) : null;
    const metadata = Promise.all([
      cachedCameras ? Promise.resolve(cachedCameras) : imagesUrl
        ? this._fetchResource(imagesUrl, "text", { signal }).then(text => this._parseCams(text))
        : Promise.resolve([]),
      camerasUrl ? this._fetchResource(camerasUrl, "text", { signal }).catch(() => "") : Promise.resolve(""),
    ]).catch(error => {
      if (current()) console.warn("Camera metadata unavailable; keeping the 3D structure:", error);
      return null;
    });

    const stream = { points: [], count: 0, capacity: 0, userNavigated: false };
    this._pointStream = stream;
    const navigate = () => { stream.userNavigated = true; };
    this.canvas?.addEventListener("pointerdown", navigate);
    this.canvas?.addEventListener("wheel", navigate, { passive: true });
    const detachNavigation = () => {
      this.canvas?.removeEventListener("pointerdown", navigate);
      this.canvas?.removeEventListener("wheel", navigate);
    };
    signal.addEventListener("abort", detachNavigation, { once: true });
    const progress = (loaded, total, done) => {
      if (!current()) return;
      const percent = done ? 100 : total > 0 ? Math.min(99, Math.floor(100 * loaded / total)) : null;
      this._setLoadingState({
        message: percent === null ? "Loading 3D points…" : `3D points · ${percent}%`,
        progress: percent === null ? 0 : percent / 100,
      });
    };
    try {
      const cached = pointsUrl ? this.parsedPointsCache.get(pointsUrl) : null;
      if (cached) {
        // Revisit cached clouds in batches too, without rebuilding a PointsCloudSystem per point.
        for (let offset = 0; offset < cached.length; offset += 16384) {
          signal.throwIfAborted();
          const batch = cached.slice(offset, offset + 16384);
          for (const point of batch) stream.points.push(point);
          this._appendPointBatch(stream, batch);
          progress(offset + batch.length, cached.length, false);
          await this._yieldFrame();
        }
      } else {
        let pending = "", batch = [], lastFlush = performance.now();
        const flush = async () => {
          signal.throwIfAborted();
          if (!batch.length) return;
          this._appendPointBatch(stream, batch);
          batch = [];
          lastFlush = performance.now();
          await this._yieldFrame();
        };
        const consume = async (text, final = false) => {
          signal.throwIfAborted();
          pending += text;
          const end = final ? pending.length : pending.lastIndexOf("\n") + 1;
          if (end <= 0) return;
          const parsed = this._parsePoints(pending.slice(0, end));
          pending = pending.slice(end);
          for (const point of parsed) {
            stream.points.push(point);
            batch.push(point);
            if (batch.length >= 16384) await flush();
          }
          if (final || (!stream.count && batch.length >= 128) || performance.now() - lastFlush >= 40) await flush();
        };
        await this._fetchResource(pointsUrl, "text", {
          signal, cache: "no-store", cacheResult: false, collectText: false,
          onProgress: progress, onText: text => consume(text),
        });
        await consume("", true);
        await flush();
      }
      signal.throwIfAborted();
      this._finishPointStream(stream);
      this.overviewPointBounds = this._robustBounds(stream.points);
      if (this.overviewPointBounds) {
        this.sceneCenter = this.overviewPointBounds.min.add(this.overviewPointBounds.max).scale(0.5);
        if (!stream.userNavigated) this._frameBounds([this.overviewPointBounds]);
      }
      if (pointsUrl) this._rememberParsed(this.parsedPointsCache, pointsUrl, stream.points, this.parsedCacheLimit);
      this._setLoadingState({ active: false });

      // Do not await metadata or photos: the structure is usable as soon as its own stream ends.
      this._sceneMetadataPromise = metadata.then(data => {
        if (!current() || !data) return;
        const [cameras, intrinsicsText] = data;
        if (imagesUrl && !cachedCameras) this._rememberParsed(this.parsedCamsCache, imagesUrl, cameras, this.parsedCacheLimit);
        const intrinsics = this._parseIntrinsics(intrinsicsText);
        this.cameras = cameras.map(camera => ({ ...camera, intrinsics: intrinsics.get(camera.cameraId) }));
        this.cameraCount = cameras.length;
        if (cameras.length) this._createFrusta(this.cameras);
        this.camIndex = 0;
        this._updateCurrentImageName();
        this._renderStats();
        void this._loadPhotos(scenePath);
      });
      return true;
    } catch (error) {
      if (current()) {
        this._finishPointStream(stream);
        this._setLoadingState({ active: true, compact: true, message: "3D point download interrupted" });
        console.warn("Unable to finish point cloud:", error);
      }
      return false;
    } finally {
      detachNavigation();
      signal.removeEventListener("abort", detachNavigation);
      if (this._pointStream === stream) this._pointStream = null;
      this._endBusy(busyToken);
    }
  }

  async loadSplatsFile({ splatsUrl, label = "", streamId = null }) {
    if (this.disposed) return false;
    // Keep a usable preview and the user's viewpoint while the next version downloads.
    const refreshing = !!(streamId && streamId === this._splatStreamId && this.mode === "splat" && this.splatsMesh);
    const busyToken = this._beginBusy("splat");
    try {
      if (!refreshing) await this._clear();
      if (this.disposed || this._busyState?.token !== busyToken) return false;
      this._splatStreamId = streamId;
      this._setIdleGroundVisible(true);
      this.mode = "splat";
      this.splatsUrl = splatsUrl;
      this.currentImageName = label;
      const token = Symbol("splatsLoad");
      this._activeSplatsToken = token;
      this._splatsMeshToken = null;
      this._applyHudMode();

      try {
        await this._ensureSplatsMesh(token);
      } catch (err) {
        if (this._activeSplatsToken === token) {
          console.warn("Failed to load gaussian splats:", err);
          this._renderStats();
        }
        return false;
      }

      if (this._activeSplatsToken !== token) {
        return false;
      }

      if (!this.splatsMesh) {
        this._renderStats();
        return false;
      }

      this.pointsCount = 0;
      this.cameraCount = 0;
      this._renderStats();

      if (refreshing) return true;

      const bb = this.splatsMesh.getBoundingInfo().boundingBox;
      const center = bb.centerWorld.clone();
      this.sceneCenter = center;
      this.camera.setTarget(center);
      const size = Math.max(
        bb.maximumWorld.x - bb.minimumWorld.x,
        bb.maximumWorld.y - bb.minimumWorld.y,
        bb.maximumWorld.z - bb.minimumWorld.z
      ) || 1.0;
      this._setZoomScale(size);
      this.camera.radius = size * 1.6;
      this.camera.rebuildAnglesAndRadius?.();
      return true;
    } finally {
      this._endBusy(busyToken);
    }
  }

  async _clear() {
    this.splatAbort?.abort();
    this._sceneLoadAbort?.abort();
    this._sceneLoadAbort = null;
    this._pointStream = null;
    this._stopPointReveal();
    this.photoGeneration += 1;
    this.photoAbort?.abort();
    this.photoAbort = null;
    for (const entry of this.photoTextures.values()) this._disposePhotoTexture(entry);
    this.photoTextures.clear();
    this.photoItems = [];
    this.photoLoading = false;
    this.photoError = "";
    this._emitPhotoState();
    this.frustumLines = [];
    this.mode = "idle";
    this.cameras = [];
    this.pointsCount = 0;
    this.cameraCount = 0;
    this.currentImageName = "";
    this._applyHudMode();

    if (this.pointsMesh) { this.pointsMesh.material?.dispose(); this.pointsMesh.dispose(); this.pointsMesh = null; }
    if (this.pcs) { this.pcs.dispose(); this.pcs = null; }
    this._clearSplats();
    this._activeSplatsToken = null;
    this._splatsMeshToken = null;
    this._splatsLoadingPromise = null;
    this.splatsUrl = null;
    this._splatStreamId = null;
    this.splatsPointCount = 0;
    this.sceneCenter = null;
    this.overviewPointBounds = null;
    this.alignmentRotation = null;
    this._updateAlignmentControl();

    if (this.frustumGroup) {
      for (const ch of this.frustumGroup.getChildren()) ch.dispose();
      this.cameraLineMeshes = [];
      this.frustumGroup.setEnabled(false);
      this.frustumGroup.setEnabled(true);
    }
    this._renderStats();
  }

  // ------------------- parsing -------------------

  _parsePoints(text) {
    const pts = [];
    for (const line of text.split("\n")) {
      if (!line || line[0] === "#") continue;
      const s = line.trim().split(/\s+/);
      if (s.length >= 7) {
        // Right-handed, Z-up reconstruction -> left-handed, Y-up viewer.
        const point = {
          x: parseFloat(s[1]), y: parseFloat(s[3]), z: parseFloat(s[2]),
          r: parseInt(s[4]) / 255, g: parseInt(s[5]) / 255, b: parseInt(s[6]) / 255
        };
        if (Object.values(point).every(Number.isFinite)) pts.push(point);
      }
    }
    return pts;
  }

  _parseCams(text) {
    const cams = [];
    const worldToViewer = this._worldToViewerMatrix();
    const localToColmap = BABYLON.Matrix.Scaling(1, -1, 1);
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = (lines[i] || "").trim();
      if (!line || line.startsWith("#")) continue;
      const s = line.split(/\s+/);
      if (s.length >= 10) {
        const qw = parseFloat(s[1]), qx = parseFloat(s[2]), qy = parseFloat(s[3]), qz = parseFloat(s[4]);
        const tx = parseFloat(s[5]), ty = parseFloat(s[6]), tz = parseFloat(s[7]);

        const qwc = new BABYLON.Quaternion(qx, qy, qz, qw);
        const Rwc = new BABYLON.Matrix(); qwc.toRotationMatrix(Rwc);
        const Rcw = Rwc.transpose();

        const t = new BABYLON.Vector3(tx, ty, tz);
        const centerWorld = BABYLON.Vector3.TransformCoordinates(t.scale(-1), Rcw);

        const center = BABYLON.Vector3.TransformCoordinates(centerWorld, worldToViewer);
        // Babylon composes row-vector transforms left to right: camera-local
        // Y-up -> COLMAP Y-down, camera -> world, then source Z-up -> viewer Y-up.
        const Rcam = localToColmap.multiply(Rcw).multiply(worldToViewer);

        const name = s.length > 9 ? s.slice(9).join(" ") : "";
        cams.push({ center, R: Rcam, name, cameraId: Number(s[8]) });

        // Skip 2D measurements line if present
        if (i + 1 < lines.length && !(lines[i + 1] || "").startsWith("#")) i++;
      }
    }
    return cams;
  }

  _worldToViewerMatrix() {
    // (x, y, z) -> (x, z, y). Swapping Y/Z also changes handedness.
    return BABYLON.Matrix.FromValues(
      1, 0, 0, 0,
      0, 0, 1, 0,
      0, 1, 0, 0,
      0, 0, 0, 1
    );
  }

  _parseIntrinsics(text) {
    const result = new Map();
    for (const line of text.split("\n")) {
      if (!line.trim() || line.startsWith("#")) continue;
      const [id, model, width, height, ...values] = line.trim().split(/\s+/);
      const p = values.map(Number);
      const singleFocal = ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"].includes(model);
      const k = { width: +width, height: +height, fx: p[0], fy: singleFocal ? p[0] : p[1], cx: p[singleFocal ? 1 : 2], cy: p[singleFocal ? 2 : 3] };
      if (Object.values(k).every(Number.isFinite) && k.fx > 0 && k.fy > 0 && k.width > 0 && k.height > 0) result.set(+id, k);
    }
    return result;
  }

  // ------------------- geometry -------------------

  _centroidSample(points, n = 100) {
    if (!points.length) return new BABYLON.Vector3(0, 0, 0);
    const step = Math.max(1, Math.floor(points.length / n));
    let x = 0, y = 0, z = 0, k = 0;
    for (let i = 0; i < points.length; i += step) {
      const p = points[i];
      x += p.x; y += p.y; z += p.z;
      k++;
      if (k >= n) break;
    }
    return new BABYLON.Vector3(x / k, y / k, z / k);
  }

  async _createPCS(points) {
    const pcs = new BABYLON.PointsCloudSystem(
      "pcs",
      { capacity: points.length, useColor: true },
      this.scene
    );

    pcs.addPoints(points.length, (particle, i) => {
      const s = points[i];
      particle.position.set(s.x, s.y, s.z);
      particle.color = new BABYLON.Color4(s.r, s.g, s.b, 1.0);
    });

    this.pointsMesh = await pcs.buildMeshAsync();

    const mat = new BABYLON.StandardMaterial("pcmat", this.scene);
    mat.pointsCloud = true;
    mat.pointSize = this.ptSize;
    mat.disableLighting = true;
    mat.useVertexColors = true;          // <-- plural in v8
    mat.emissiveColor = new BABYLON.Color3(1, 1, 1); // ensure full vertex color visibility
    this.pointsMesh.material = mat;
    this.pcs = pcs;
  }

  _appendPointBatch(stream, points) {
    if (!points.length || this._pointStream !== stream) return;
    const oldCount = stream.count;
    const count = oldCount + points.length;
    const grow = count > stream.capacity;
    if (grow) {
      const capacity = Math.max(4096, 2 ** Math.ceil(Math.log2(count)));
      for (const [key, width] of [["positions", 3], ["colors", 4], ["arrivals", 1]]) {
        const buffer = new Float32Array(capacity * width);
        if (stream[key]) buffer.set(stream[key]);
        stream[key] = buffer;
      }
      stream.capacity = capacity;
    }
    const arrival = this._pointReveal ? (performance.now() - this._pointReveal.start) / 1200 : 0;
    for (let i = 0; i < points.length; i++) {
      const p = points[i], index = oldCount + i;
      stream.positions.set([p.x, p.y, p.z], index * 3);
      stream.colors.set([p.r, p.g, p.b, 1], index * 4);
      stream.arrivals[index] = arrival;
    }
    const first = !this.pointsMesh;
    if (first) {
      this.pointsMesh = new BABYLON.Mesh("streamed-points", this.scene);
      const material = new BABYLON.StandardMaterial("pcmat", this.scene);
      material.pointsCloud = true;
      material.pointSize = this.ptSize;
      material.disableLighting = true;
      material.useVertexColors = true;
      material.emissiveColor = BABYLON.Color3.White();
      this.pointsMesh.material = material;
    }
    const mesh = this.pointsMesh;
    for (const [kind, key, width] of [["position", "positions", 3], ["color", "colors", 4], ["arrival", "arrivals", 1]]) {
      if (grow) mesh.setVerticesData(kind, stream[key], true, width);
      else mesh.getVertexBuffer(kind)?.updateDirectly(stream[key].subarray(oldCount * width, count * width), oldCount * width);
    }
    if (grow) mesh.setIndices(Uint32Array.from({ length: stream.capacity }, (_, i) => i));
    mesh.subMeshes[0].indexCount = count;
    mesh.subMeshes[0].verticesCount = count;
    stream.count = count;
    this.pointsCount = count;
    // Sample across the received structure, rather than repeatedly sorting the whole cloud.
    const sample = [];
    const step = Math.max(1, Math.floor(count / 2048));
    for (let i = 0; i < count; i += step) sample.push(stream.points[i]);
    this.overviewPointBounds = this._robustBounds(sample);
    if (this.overviewPointBounds) {
      this.sceneCenter = this.overviewPointBounds.min.add(this.overviewPointBounds.max).scale(0.5);
      if (!stream.userNavigated) this._frameBounds([this.overviewPointBounds]);
    }
    if (first) this._startPointReveal(true);
    if (this._pointReveal) this._pointReveal.lastArrival = arrival;
    mesh.alwaysSelectAsActiveMesh = true;
    this._renderStats();
  }

  _finishPointStream(stream) {
    if (stream.finished || this._pointStream !== stream) return;
    stream.finished = true;
    const mesh = this.pointsMesh;
    if (!mesh) return;
    // One final compaction leaves exact CPU vertices available for Z-up alignment and export.
    mesh.setVerticesData("position", stream.positions.slice(0, stream.count * 3), true);
    mesh.setVerticesData("color", stream.colors.slice(0, stream.count * 4), true);
    mesh.setIndices(Uint32Array.from({ length: stream.count }, (_, i) => i));
    if (this._pointReveal) {
      mesh.setVerticesData("arrival", stream.arrivals.slice(0, stream.count), false, 1);
      this._pointReveal.streaming = false;
    } else {
      mesh.removeVerticesData("arrival");
      mesh.alwaysSelectAsActiveMesh = false;
    }
    stream.positions = stream.colors = stream.arrivals = null;
    mesh.refreshBoundingInfo();
  }

  _startPointReveal(streaming = false) {
    this._stopPointReveal();
    const mesh = this.pointsMesh;
    if (!mesh || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    // Animate only the rendered positions. The original vertices, bounds and cached points
    // stay untouched, with one uniform update per frame regardless of cloud size.
    const material = new BABYLON.ShaderMaterial("point-reveal", this.scene, {
      vertexSource: `precision highp float;
        attribute vec3 position;
        attribute vec4 color;
        attribute float arrival;
        uniform mat4 worldViewProjection;
        uniform vec3 cloudCenter;
        uniform float cloudScale, flyProgress, pointSize;
        varying vec4 vColor;
        float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
        void main() {
          vec3 p = (position - cloudCenter) / cloudScale;
          float seed = hash(p);
          vec3 direction = normalize(vec3(seed, hash(p + 3.1), hash(p + 7.7)) - 0.5 + vec3(0.0001));
          float t = clamp((flyProgress - arrival - seed * 0.2) / 0.8, 0.0, 1.0);
          float eased = 1.0 - pow(1.0 - t, 3.0);
          vec3 start = cloudCenter + direction * cloudScale * (1.5 + seed);
          vec3 arc = vec3(-direction.z, direction.x * 0.4, direction.x) * sin(t * 3.14159265) * cloudScale * 0.3;
          gl_Position = worldViewProjection * vec4(mix(start, position, eased) + arc, 1.0);
          gl_PointSize = pointSize * (1.0 + 0.35 * (1.0 - eased));
          vColor = vec4(mix(vec3(0.92, 0.75, 0.48), color.rgb, eased), color.a);
        }`,
      fragmentSource: `precision highp float;
        varying vec4 vColor;
        void main() { gl_FragColor = vColor; }`,
    }, {
      attributes: ["position", "color", "arrival"],
      uniforms: ["worldViewProjection", "cloudCenter", "cloudScale", "flyProgress", "pointSize"],
    });
    material.pointsCloud = true;
    material.setVector3("cloudCenter", this.sceneCenter || BABYLON.Vector3.Zero());
    material.setFloat("cloudScale", Math.max(1e-8, this._sceneExtent()));
    material.setFloat("flyProgress", 0);
    material.setFloat("pointSize", this.ptSize);
    if (!mesh.isVerticesDataPresent("arrival")) mesh.setVerticesData("arrival", new Float32Array(mesh.getTotalVertices()), false, 1);
    const reveal = {
      mesh, material, original: mesh.material, alwaysActive: mesh.alwaysSelectAsActiveMesh,
      start: performance.now(), lastArrival: 0, streaming,
    };
    this._pointReveal = reveal;
    mesh.material = material;
    mesh.alwaysSelectAsActiveMesh = true;
    material.onError = () => this._stopPointReveal();
    reveal.observer = this.scene.onBeforeRenderObservable.add(() => {
      if (!material.isReady(mesh)) return;
      const now = performance.now();
      const progress = (now - reveal.start) / 1200;
      material.setFloat("flyProgress", progress);
      material.setFloat("pointSize", this.ptSize);
      if (!reveal.streaming && progress >= reveal.lastArrival + 1) this._stopPointReveal();
    });
  }

  _stopPointReveal() {
    const reveal = this._pointReveal;
    if (!reveal) return;
    this._pointReveal = null;
    this.scene.onBeforeRenderObservable.remove(reveal.observer);
    reveal.mesh.material = reveal.original;
    reveal.mesh.alwaysSelectAsActiveMesh = reveal.alwaysActive;
    reveal.mesh.removeVerticesData("arrival");
    reveal.material.dispose();
  }

  _clearSplats() {
    if (this.splatsMesh) {
      this.splatsMesh.dispose();
      this.splatsMesh = null;
    }
    this._splatsLoadingPromise = null;
    this._splatsMeshToken = null;
    this.splatsPointCount = 0;
  }

  async _ensureSplatsMesh(requestToken = null) {
    const token = requestToken ?? this._activeSplatsToken;
    if (!this.splatsUrl || !token) return;
    if (this.splatsMesh && this._splatsMeshToken === token) return;

    if (this._splatsLoadingPromise?.token === token) {
      await this._splatsLoadingPromise.promise;
      return;
    }

    const isTokenActive = () => this._activeSplatsToken === token && this.mode === "splat";
    const updateLoading = (payload) => {
      if (isTokenActive()) this._setLoadingState(payload);
    };

    updateLoading({ active: true, compact: true, message: "Updating splat view…", progress: 0 });

    this.splatAbort?.abort();
    const controller = new AbortController();
    this.splatAbort = controller;
    const promise = (async () => {
      const buffer = await this._fetchResource(this.splatsUrl, "arrayBuffer", {
        signal: controller.signal,
        cache: "no-store",
        cacheResult: false,
      });
      if (!isTokenActive()) return;

      if (typeof BABYLON.GaussianSplattingMesh !== "function") {
        throw new Error("This browser build does not include the Gaussian splat renderer.");
      }

      updateLoading({ message: "Converting Gaussian splats…", progress: 0.15 });
      const converted = await BABYLON.GaussianSplattingMesh.ConvertPLYWithSHToSplatAsync(buffer);
      if (!isTokenActive()) return;

      if (!converted?.buffer?.byteLength) {
        throw new Error("The Gaussian splat file contained no renderable data.");
      }

      updateLoading({ message: "Preparing Gaussian renderer…", progress: 0.7 });
      await this._createSplatsMesh(converted, token);

      if (!isTokenActive()) return;

      this.splatsPointCount = this.splatsMesh?.getTotalVertices?.() ?? 0;
      this._renderStats();
      updateLoading({ progress: 1 });
    })();

    this._splatsLoadingPromise = { token, promise };

    try {
      await promise;
    } finally {
      if (this._splatsLoadingPromise?.token === token) {
        this._splatsLoadingPromise = null;
      }
      updateLoading({ active: false });
    }
  }

  async _createSplatsMesh(converted, token = null) {
    const isTokenActive = () => !token || (this._activeSplatsToken === token && this.mode === "splat");
    if (!isTokenActive()) return;

    // Babylon rasterizes the full anisotropic covariance in screen space, sorts
    // translucent splats by depth, and evaluates any spherical-harmonic color
    // coefficients included in the PLY. This is intentionally not a polygonal
    // ellipsoid approximation.
    const mesh = new BABYLON.GaussianSplattingMesh(
      `gaussian-splats-${Date.now()}`,
      null,
      this.scene,
      false
    );
    mesh.isPickable = false;
    mesh.setEnabled(false);
    // Transform the complete splat frame, including anisotropic covariance.
    mesh.rotationQuaternion = new BABYLON.Quaternion();
    this._worldToViewerMatrix().decompose(mesh.scaling, mesh.rotationQuaternion, mesh.position);

    try {
      await mesh.updateDataAsync(converted.buffer, converted.sh ?? undefined);
    } catch (err) {
      mesh.dispose();
      throw err;
    }

    if (!isTokenActive()) {
      mesh.dispose();
      return;
    }

    mesh.computeWorldMatrix(true);
    mesh.refreshBoundingInfo();
    const previousMesh = this.splatsMesh;
    this.splatsMesh = mesh;
    this._splatsMeshToken = token ?? null;
    mesh.setEnabled(true);
    previousMesh?.dispose();
  }

  _parseSplatsPly(buffer) {
    if (typeof buffer === "string") {
      buffer = new TextEncoder().encode(buffer).buffer;
    }
    if (!(buffer instanceof ArrayBuffer)) {
      console.warn("PLY parser expected ArrayBuffer input.");
      return [];
    }

    const bytes = new Uint8Array(buffer);
    if (bytes.length < 32) {
      console.warn("PLY file too small.");
      return [];
    }

    const findHeaderEnd = () => {
      const marker = "end_header";
      const markerChars = Array.from(marker).map(ch => ch.charCodeAt(0));
      for (let i = 0; i <= bytes.length - markerChars.length; i++) {
        let matched = true;
        for (let j = 0; j < markerChars.length; j++) {
          if (bytes[i + j] !== markerChars[j]) {
            matched = false;
            break;
          }
        }
        if (!matched) continue;
        let end = i + markerChars.length;
        if (end < bytes.length && bytes[end] === 13) end += 1;
        if (end < bytes.length && bytes[end] === 10) {
          return end + 1;
        }
        if (end < bytes.length && bytes[end] === 13 && end + 1 < bytes.length && bytes[end + 1] === 10) {
          return end + 2;
        }
      }
      return -1;
    };

    const headerEnd = findHeaderEnd();
    if (headerEnd < 0) {
      console.warn("PLY header missing end_header.");
      return [];
    }

    const headerText = new TextDecoder().decode(bytes.slice(0, headerEnd));
    const headerLines = headerText.split(/\r?\n/).map(line => line.trim());
    if (!headerLines.length || headerLines[0].toLowerCase() !== "ply") {
      console.warn("gaussian_splats.ply missing PLY header.");
      return [];
    }

    let formatLine = null;
    let format = "";
    let vertexCount = 0;
    const properties = [];
    let inVertexElement = false;
    for (let i = 1; i < headerLines.length; i++) {
      const line = headerLines[i];
      if (!line) continue;
      if (line.startsWith("format")) {
        formatLine = line;
        const parts = line.split(/\s+/);
        format = (parts[1] || "").toLowerCase();
        continue;
      }
      if (line.startsWith("element")) {
        const parts = line.split(/\s+/);
        inVertexElement = parts[1] === "vertex";
        if (inVertexElement && parts.length >= 3) {
          vertexCount = parseInt(parts[2], 10) || 0;
        }
        continue;
      }
      if (line.startsWith("property")) {
        if (!inVertexElement) continue;
        const parts = line.split(/\s+/);
        if (parts[1] === "list") {
          console.warn("List properties in vertex element are not supported.");
          return [];
        }
        const type = parts[1];
        const name = parts[parts.length - 1];
        const size = this._plyTypeSize(type);
        if (!size) {
          console.warn(`Unsupported PLY property type: ${type}`);
          return [];
        }
        properties.push({ type, name, size });
        continue;
      }
    }

    if (!format) {
      console.warn("PLY header missing format line.");
      return [];
    }
    if (!properties.length) {
      console.warn("PLY vertex element missing properties.");
      return [];
    }

    const payloadBuffer = buffer.slice(headerEnd);
    if (format.includes("ascii")) {
      const payloadText = new TextDecoder().decode(payloadBuffer);
      return this._parseSplatsPlyAscii(payloadText, properties, vertexCount);
    }
    if (format.includes("binary_little_endian")) {
      return this._parseSplatsPlyBinary(payloadBuffer, properties, vertexCount, true);
    }
    if (format.includes("binary_big_endian")) {
      return this._parseSplatsPlyBinary(payloadBuffer, properties, vertexCount, false);
    }

    console.warn(`Unsupported PLY format: ${formatLine || format}`);
    return [];
  }

  _parseSplatsPlyAscii(text, properties, vertexCount) {
    const lines = text.split(/\r?\n/);
    const idx = this._plyBuildIndexMap(properties);
    if (idx.x < 0 || idx.y < 0 || idx.z < 0) {
      console.warn("PLY file does not define x/y/z properties.");
      return [];
    }
    const points = [];
    const totalVertices = vertexCount || lines.length;
    for (let i = 0; i < lines.length; i++) {
      if (vertexCount && points.length >= vertexCount) break;
      const line = (lines[i] || "").trim();
      if (!line || line.startsWith("comment")) continue;
      const parts = line.split(/\s+/);
      if (parts.length < properties.length) continue;
      const x = parseFloat(parts[idx.x]);
      const y = parseFloat(parts[idx.y]);
      const z = parseFloat(parts[idx.z]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

      const components = {};
      if (idx.red >= 0) components.red = parseFloat(parts[idx.red]);
      if (idx.green >= 0) components.green = parseFloat(parts[idx.green]);
      if (idx.blue >= 0) components.blue = parseFloat(parts[idx.blue]);
      if (idx.fdc0 >= 0) components.fdc0 = parseFloat(parts[idx.fdc0]);
      if (idx.fdc1 >= 0) components.fdc1 = parseFloat(parts[idx.fdc1]);
      if (idx.fdc2 >= 0) components.fdc2 = parseFloat(parts[idx.fdc2]);
      const { r, g, b } = this._plyColorFromComponents(components);

      const opacityRaw = idx.opacity >= 0 ? parseFloat(parts[idx.opacity]) : 1.0;
      const opacity = this._plyOpacity(opacityRaw, idx);
      const scale0 = idx.scale0 >= 0 ? parseFloat(parts[idx.scale0]) : 0;
      const scale1 = idx.scale1 >= 0 ? parseFloat(parts[idx.scale1]) : 0;
      const scale2 = idx.scale2 >= 0 ? parseFloat(parts[idx.scale2]) : 0;
      const rot0 = idx.rot0 >= 0 ? parseFloat(parts[idx.rot0]) : 0;
      const rot1 = idx.rot1 >= 0 ? parseFloat(parts[idx.rot1]) : 0;
      const rot2 = idx.rot2 >= 0 ? parseFloat(parts[idx.rot2]) : 0;
      const rot3 = idx.rot3 >= 0 ? parseFloat(parts[idx.rot3]) : 1;

      points.push({
        x,
        y: z,
        z: y,
        r,
        g,
        b,
        opacity,
        scale: [scale0, scale1, scale2],
        rotation: this._plyRotation(rot0, rot1, rot2, rot3, idx),
      });
      if (points.length >= totalVertices) break;
    }
    return this._plyDownsample(points);
  }

  _parseSplatsPlyBinary(buffer, properties, vertexCount, littleEndian) {
    if (!(buffer instanceof ArrayBuffer)) {
      if (ArrayBuffer.isView(buffer)) {
        buffer = buffer.buffer;
      } else {
        console.warn("Binary PLY parser expected ArrayBuffer.");
        return [];
      }
    }
    const idx = this._plyBuildIndexMap(properties);
    if (idx.x < 0 || idx.y < 0 || idx.z < 0) {
      console.warn("PLY file does not define x/y/z properties.");
      return [];
    }

    const view = new DataView(buffer);
    const stride = properties.reduce((sum, prop) => sum + prop.size, 0);
    if (!stride) {
      console.warn("Invalid stride computed for PLY vertex data.");
      return [];
    }
    const availableVertices = Math.floor(view.byteLength / stride);
    const totalVertices = vertexCount ? Math.min(vertexCount, availableVertices) : availableVertices;
    const points = [];
    let offset = 0;
    for (let i = 0; i < totalVertices; i++) {
      let x, y, z;
      let red, green, blue;
      let fdc0, fdc1, fdc2;
      let opacity = 1.0;
      let scale0 = 0, scale1 = 0, scale2 = 0;
      let rot0 = 0, rot1 = 0, rot2 = 0, rot3 = 1;
      for (const prop of properties) {
        const value = this._plyReadValue(view, offset, prop.type, littleEndian);
        offset += prop.size;
        switch (prop.name) {
          case "x": x = value; break;
          case "y": y = value; break;
          case "z": z = value; break;
          case "red": red = value; break;
          case "green": green = value; break;
          case "blue": blue = value; break;
          case "f_dc_0": fdc0 = value; break;
          case "f_dc_1": fdc1 = value; break;
          case "f_dc_2": fdc2 = value; break;
          case "opacity": opacity = value; break;
          case "scale_0": scale0 = value; break;
          case "scale_1": scale1 = value; break;
          case "scale_2": scale2 = value; break;
          case "rot_0": rot0 = value; break;
          case "rot_1": rot1 = value; break;
          case "rot_2": rot2 = value; break;
          case "rot_3": rot3 = value; break;
          default: break;
        }
      }
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const { r, g, b } = this._plyColorFromComponents({ red, green, blue, fdc0, fdc1, fdc2 });
      const alpha = this._plyOpacity(opacity, idx);
      points.push({
        x,
        y: z,
        z: y,
        r,
        g,
        b,
        opacity: alpha,
        scale: [scale0, scale1, scale2],
        rotation: this._plyRotation(rot0, rot1, rot2, rot3, idx),
      });
    }
    return this._plyDownsample(points);
  }

  _plyBuildIndexMap(properties) {
    const find = (name) => properties.findIndex(p => p.name === name);
    return {
      x: find("x"),
      y: find("y"),
      z: find("z"),
      red: find("red"),
      green: find("green"),
      blue: find("blue"),
      fdc0: find("f_dc_0"),
      fdc1: find("f_dc_1"),
      fdc2: find("f_dc_2"),
      opacity: find("opacity"),
      scale0: find("scale_0"),
      scale1: find("scale_1"),
      scale2: find("scale_2"),
      rot0: find("rot_0"),
      rot1: find("rot_1"),
      rot2: find("rot_2"),
      rot3: find("rot_3"),
    };
  }

  _plyOpacity(value, indexMap) {
    if (!Number.isFinite(value)) return 1.0;
    const isGaussianPly = indexMap.fdc0 >= 0 && indexMap.scale0 >= 0 && indexMap.rot0 >= 0;
    if (isGaussianPly) {
      // Gsplat and the original 3DGS PLY format store opacity as a logit.
      return 1 / (1 + Math.exp(-value));
    }
    return Math.min(1, Math.max(0, value));
  }

  _plyRotation(rot0, rot1, rot2, rot3, indexMap) {
    const hasGaussianRotation = indexMap.rot0 >= 0 && indexMap.rot1 >= 0 && indexMap.rot2 >= 0 && indexMap.rot3 >= 0;
    if (!hasGaussianRotation) return [0, 0, 0, 1];
    // Gsplat writes quaternions as w, x, y, z; Babylon consumes x, y, z, w.
    return [rot1, rot2, rot3, rot0];
  }

  _plyColorFromComponents(components) {
    const clamp01 = (v) => Math.min(1, Math.max(0, v));
    const { red, green, blue, fdc0, fdc1, fdc2 } = components;
    if ([red, green, blue].every(v => typeof v === "number" && Number.isFinite(v))) {
      return {
        r: clamp01(red / 255),
        g: clamp01(green / 255),
        b: clamp01(blue / 255),
      };
    }
    if ([fdc0, fdc1, fdc2].every(v => typeof v === "number" && Number.isFinite(v))) {
      const SH_C0 = 0.28209479177387814;
      return {
        r: clamp01(0.5 + SH_C0 * fdc0),
        g: clamp01(0.5 + SH_C0 * fdc1),
        b: clamp01(0.5 + SH_C0 * fdc2),
      };
    }
    return { r: 0.6, g: 0.6, b: 0.6 };
  }

  _plyDownsample(points, maxPoints = 200000) {
    if (points.length <= maxPoints) return points;
    const step = Math.ceil(points.length / maxPoints);
    const sampled = [];
    for (let i = 0; i < points.length; i += step) {
      sampled.push(points[i]);
    }
    return sampled;
  }

  _plyTypeSize(type) {
    const key = (type || "").toLowerCase();
    switch (key) {
      case "char":
      case "uchar":
      case "int8":
      case "uint8":
        return 1;
      case "short":
      case "ushort":
      case "int16":
      case "uint16":
        return 2;
      case "int":
      case "uint":
      case "float":
      case "float32":
      case "int32":
      case "uint32":
        return 4;
      case "double":
      case "float64":
      case "int64":
      case "uint64":
        return 8;
      default:
        return 0;
    }
  }

  _plyReadValue(view, offset, type, littleEndian) {
    const key = (type || "").toLowerCase();
    switch (key) {
      case "char":
      case "int8":
        return view.getInt8(offset);
      case "uchar":
      case "uint8":
        return view.getUint8(offset);
      case "short":
      case "int16":
        return view.getInt16(offset, littleEndian);
      case "ushort":
      case "uint16":
        return view.getUint16(offset, littleEndian);
      case "int":
      case "int32":
        return view.getInt32(offset, littleEndian);
      case "uint":
      case "uint32":
        return view.getUint32(offset, littleEndian);
      case "float":
      case "float32":
        return view.getFloat32(offset, littleEndian);
      case "double":
      case "float64":
        return view.getFloat64(offset, littleEndian);
      case "int64":
      case "uint64":
        {
          let high;
          let low;
          if (littleEndian) {
            low = view.getUint32(offset, true);
            high = view.getInt32(offset + 4, true);
          } else {
            high = view.getInt32(offset, false);
            low = view.getUint32(offset + 4, false);
          }
          return high * 2 ** 32 + low;
        }
      default:
        return 0;
    }
  }

  async _fetchResource(url, kind = "text", fetchOpts = {}) {
    if (!url) {
      return kind === "arrayBuffer" ? new ArrayBuffer(0) : "";
    }
    const { cacheResult = true, onProgress, onText, collectText = true, ...requestInit } = fetchOpts;
    const key = `${kind}:${url}`;
    if (cacheResult && this.resourceCache.has(key)) {
      const cached = this.resourceCache.get(key);
      if (kind === "text") await onText?.(cached);
      onProgress?.(1, 1, true);
      if (kind === "arrayBuffer") {
        return cached.slice(0);
      }
      return cached;
    }
    const response = await fetch(url, requestInit);
    if (!response.ok) {
      throw new Error(`Failed to load ${url}`);
    }
    if (kind === "arrayBuffer") {
      const buffer = await response.arrayBuffer();
      if (!cacheResult) return buffer;
      this._rememberResource(key, buffer);
      return buffer.slice(0);
    }
    let text;
    if ((onProgress || onText) && response.body?.getReader) {
      // Content-Length is only comparable to decoded bytes for uncompressed responses.
      const length = response.headers.get("content-encoding") ? null : Number(response.headers.get("content-length")) || null;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const chunks = [];
      let loaded = 0;
      onProgress?.(0, length);
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          loaded += value.byteLength;
          const decoded = decoder.decode(value, { stream: true });
          if (collectText) chunks.push(decoded);
          await onText?.(decoded);
          onProgress?.(loaded, length);
        }
        const tail = decoder.decode();
        if (collectText) chunks.push(tail);
        await onText?.(tail);
      } finally {
        reader.releaseLock();
      }
      text = chunks.join("");
      onProgress?.(loaded, loaded, true);
    } else {
      text = await response.text();
      await onText?.(text);
      onProgress?.(1, 1, true);
    }
    if (cacheResult) this._rememberResource(key, text);
    return text;
  }

  _rememberResource(key, value) {
    this.resourceCache.set(key, value);
    if (this.resourceCache.size > this.resourceCacheLimit) {
      const firstKey = this.resourceCache.keys().next().value;
      this.resourceCache.delete(firstKey);
    }
  }

  _rememberParsed(cache, key, value, limit) {
    if (!cache || !key) return;
    cache.set(key, value);
    while (cache.size > limit) {
      const oldestKey = cache.keys().next().value;
      if (oldestKey === key && cache.size === 1) break;
      cache.delete(oldestKey);
    }
  }

  _sceneExtent() {
    if (this.overviewPointBounds) {
      const size = this.overviewPointBounds.max.subtract(this.overviewPointBounds.min);
      return Math.max(size.x, size.y, size.z) || 1.0;
    }
    if (!this.pointsMesh) return 1.0;
    const bb = this.pointsMesh.getBoundingInfo().boundingBox;
    const size = bb.maximumWorld.subtract(bb.minimumWorld);
    return Math.max(size.x, size.y, size.z) || 1.0;
  }

  _imageCorners(camera, depth) {
    const k = camera.intrinsics ?? { width: 4, height: 3, fx: 5, fy: 5, cx: 2, cy: 1.5 };
    return [[0, 0], [k.width, 0], [k.width, k.height], [0, k.height]].map(([x, y]) =>
      new BABYLON.Vector3((x - k.cx) / k.fx * depth, -(y - k.cy) / k.fy * depth, depth));
  }

  _createFrusta(cams) {
    const size = this.pointsMesh ? this._sceneExtent() : cams.reduce((size, camera) => Math.max(size, camera.center.length()), 1);
    this.frustumDepth = size * 0.04;
    this.frustumLines = cams.map((camera) => {
      const corners = this._imageCorners(camera, this.frustumDepth).map(point =>
        BABYLON.Vector3.TransformNormal(point, camera.R).add(camera.center));
      return corners.flatMap((point, index) => [[camera.center, point], [point, corners[(index + 1) % 4]]]);
    });
    const mesh = BABYLON.MeshBuilder.CreateLineSystem("camera-wireframes", { lines: this.frustumLines.flat(), updatable: true }, this.scene);
    mesh.parent = this.frustumGroup;
    mesh.isPickable = false;
    mesh.alwaysSelectAsActiveMesh = true;
    mesh.freezeWorldMatrix();
    this.cameraLineMeshes = [mesh];
    this.setOverlayTheme(this.overlayTheme);
    this.setShowCams(this.showCams);
  }

  getPhotoState() {
    return { items: this.photoItems, selected: this.camIndex ?? 0, loading: this.photoLoading, error: this.photoError };
  }

  subscribePhotos(listener) {
    this.photoListeners.add(listener);
    return () => this.photoListeners.delete(listener);
  }

  _emitPhotoState() {
    for (const listener of this.photoListeners ?? []) listener(this.getPhotoState());
  }

  async _loadPhotos(scenePath) {
    this.photoItems = this.cameras.map(camera => ({ name: camera.name, thumbnail: null, preview: null }));
    this.photoLoading = Boolean(scenePath);
    this._emitPhotoState();
    if (!scenePath || !this.cameras.length) return;
    const controller = new AbortController();
    this.photoAbort = controller;
    try {
      const response = await fetch('/api/photos?' + new URLSearchParams({ scene: scenePath }), { signal: controller.signal });
      if (!response.ok) throw new Error("Source images could not be located.");
      const data = await response.json();
      if (controller.signal.aborted) return;
      const byName = new Map(data.items.map(item => [item.name, item]));
      this.photoItems = this.cameras.map(camera => byName.get(camera.name) ?? { name: camera.name, thumbnail: null, preview: null });
      if (!this.photoItems.some(item => item.thumbnail)) this.photoError = "Source images unavailable. Restore the dataset to its saved location.";
      this._updatePhotoTextures();
    } catch (error) {
      if (controller.signal.aborted) return;
      this.photoError = "Source images unavailable. The 3D reconstruction is still usable.";
      console.warn("Unable to load scene photos", error);
    } finally {
      if (!controller.signal.aborted) { this.photoLoading = false; this._emitPhotoState(); }
    }
  }

  _disposePhotoTexture(entry) {
    entry.mesh?.dispose();
    entry.material?.dispose();
    entry.texture?.dispose();
  }

  _updatePhotoTextures() {
    if (!this.photoItems.length || !this.cameras.length || document.hidden) return;
    const selected = this.cameras[this.camIndex];
    const hiddenCamera = selected && BABYLON.Vector3.DistanceSquared(this.camera.position, selected.center) < this.frustumDepth ** 2 ? this.camIndex : -1;
    if (this.hiddenCamera !== hiddenCamera) { this._syncCameraVisuals(); return; }
    const planes = BABYLON.Frustum.GetPlanes(this.camera.getTransformationMatrix());
    const candidates = [];
    if (this.showCams) {
      for (let index = 0; index < this.cameras.length; index++) {
        if (!this.photoItems[index]?.thumbnail) continue;
        const camera = this.cameras[index];
        const distance = BABYLON.Vector3.DistanceSquared(camera.center, this.camera.position);
        const radius = this.frustumDepth * 2;
        // Nearby photo planes must not cover the reconstruction when looking
        // through one of several closely spaced source cameras.
        if (distance < radius * radius * 4) continue;
        if (planes.some(plane => plane.dotCoordinate(camera.center) < -radius)) continue;
        // Do not fetch textures for frustums smaller than a few screen pixels.
        const projected = radius * this.engine.getRenderHeight() / Math.max(Math.sqrt(distance), 0.001);
        if (projected < 3) continue;
        candidates.push({ index, distance });
      }
    }
    candidates.sort((a, b) => a.distance - b.distance);
    const wanted = new Set(candidates.slice(0, this.photoVisibleLimit).map(item => item.index));
    for (const [index, entry] of this.photoTextures) {
      entry.mesh?.setEnabled(wanted.has(index));
      if (wanted.has(index)) entry.used = performance.now();
    }
    let loading = [...this.photoTextures.values()].filter(entry => entry.loading).length;
    for (const index of wanted) {
      if (loading >= 4) break;
      if (this.photoTextures.has(index)) continue;
      if (this.photoTextures.size >= this.photoTextureLimit) {
        const evict = [...this.photoTextures.entries()].filter(([key]) => !wanted.has(key)).sort((a, b) => a[1].used - b[1].used)[0];
        if (!evict) break;
        this._disposePhotoTexture(evict[1]);
        this.photoTextures.delete(evict[0]);
      }
      const generation = this.photoGeneration;
      const entry = { loading: true, used: performance.now(), texture: null, mesh: null, material: null };
      this.photoTextures.set(index, entry);
      loading++;
      entry.texture = new BABYLON.Texture(this.photoItems[index].thumbnail, this.scene, false, true, BABYLON.Texture.TRILINEAR_SAMPLINGMODE, () => queueMicrotask(() => {
        if (generation !== this.photoGeneration || this.photoTextures.get(index) !== entry) return;
        entry.loading = false;
        this._createPhotoPlane(index, entry);
        this._updatePhotoTextures();
      }), () => {
        if (generation !== this.photoGeneration || this.photoTextures.get(index) !== entry) return;
        entry.loading = false;
        entry.texture?.dispose();
        entry.texture = null;
      });
    }
  }

  _createPhotoPlane(index, entry) {
    const camera = this.cameras[index];
    const corners = this._imageCorners(camera, this.frustumDepth);
    const mesh = new BABYLON.Mesh("camera-photo-" + index, this.scene);
    const data = new BABYLON.VertexData();
    data.positions = corners.flatMap(point => [point.x, point.y, point.z]);
    data.indices = [0, 1, 2, 0, 2, 3];
    data.uvs = [0, 1, 1, 1, 1, 0, 0, 0];
    data.normals = [0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1];
    data.applyToMesh(mesh);
    mesh.position.copyFrom(camera.center);
    mesh.rotationQuaternion = BABYLON.Quaternion.FromRotationMatrix(camera.R);
    mesh.parent = this.frustumGroup;
    mesh.metadata = { cameraIndex: index };
    const material = new BABYLON.StandardMaterial("photo-material-" + index, this.scene);
    material.disableLighting = true;
    material.backFaceCulling = false;
    // StandardMaterial adds emissiveColor to emissiveTexture; white would
    // saturate every photo to a blank white rectangle.
    material.emissiveColor = BABYLON.Color3.Black();
    material.emissiveTexture = entry.texture;
    material.diffuseColor = BABYLON.Color3.Black();
    material.specularColor = BABYLON.Color3.Black();
    mesh.material = material;
    material.freeze();
    mesh.freezeWorldMatrix();
    entry.mesh = mesh;
    entry.material = material;
  }

  // ------------------- camera framing -------------------

  _cameraPlaneRotation() {
    const cameras = this.cameras.filter(camera => camera.center.asArray().every(Number.isFinite));
    if (cameras.length < 3) throw new Error("At least three camera centers are needed to fit a plane.");
    const center = cameras.reduce((sum, camera) => sum.add(camera.center), BABYLON.Vector3.Zero()).scale(1 / cameras.length);
    const offsets = cameras.map(camera => camera.center.subtract(center));
    const scale = Math.max(...offsets.map(offset => offset.length()));
    if (!scale) throw new Error("Camera centers coincide; a plane cannot be estimated.");
    const covariance = Array.from({ length: 3 }, () => [0, 0, 0]);
    for (const offset of offsets) {
      const xyz = offset.scale(1 / scale).asArray();
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) covariance[i][j] += xyz[i] * xyz[j] / cameras.length;
    }
    // Jacobi diagonalization of a symmetric 3x3 covariance. The eigenvector
    // with the smallest eigenvalue is the orthogonal least-squares plane normal.
    const vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let iteration = 0; iteration < 32; iteration++) {
      let p = 0, q = 1;
      for (const [i, j] of [[0, 2], [1, 2]]) {
        if (Math.abs(covariance[i][j]) > Math.abs(covariance[p][q])) { p = i; q = j; }
      }
      if (Math.abs(covariance[p][q]) < 1e-12) break;
      const angle = 0.5 * Math.atan2(2 * covariance[p][q], covariance[q][q] - covariance[p][p]);
      const c = Math.cos(angle), s = Math.sin(angle);
      const pp = covariance[p][p], qq = covariance[q][q], pq = covariance[p][q];
      covariance[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq;
      covariance[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq;
      covariance[p][q] = covariance[q][p] = 0;
      for (let k = 0; k < 3; k++) {
        if (k !== p && k !== q) {
          const kp = covariance[k][p], kq = covariance[k][q];
          covariance[k][p] = covariance[p][k] = c * kp - s * kq;
          covariance[k][q] = covariance[q][k] = s * kp + c * kq;
        }
        const vp = vectors[k][p], vq = vectors[k][q];
        vectors[k][p] = c * vp - s * vq;
        vectors[k][q] = s * vp + c * vq;
      }
    }
    const order = [0, 1, 2].sort((i, j) => covariance[i][i] - covariance[j][j]);
    if (covariance[order[1]][order[1]] <= covariance[order[2]][order[2]] * 1e-6 ||
        covariance[order[0]][order[0]] > covariance[order[1]][order[1]] * 0.9) {
      throw new Error("Camera centers do not define a stable plane (too linear or too widely scattered).");
    }
    let normal = new BABYLON.Vector3(...vectors.map(row => row[order[0]])).normalize();
    const up = BABYLON.Vector3.Up(); // Viewer Y corresponds to reconstruction Z.
    const cameraUp = cameras.reduce((sum, camera) => sum.add(BABYLON.Vector3.TransformNormal(up, camera.R)), BABYLON.Vector3.Zero());
    const sign = Math.abs(BABYLON.Vector3.Dot(normal, cameraUp)) > cameraUp.length() * 1e-6
      ? BABYLON.Vector3.Dot(normal, cameraUp) : BABYLON.Vector3.Dot(normal, up);
    if (sign < 0) normal = normal.negate();
    const cosine = Math.max(-1, Math.min(1, BABYLON.Vector3.Dot(normal, up)));
    if (cosine > 1 - 1e-12) return BABYLON.Matrix.Identity();
    const axis = BABYLON.Vector3.Cross(normal, up);
    if (axis.lengthSquared() < 1e-12) axis.copyFrom(BABYLON.Vector3.Right());
    return BABYLON.Matrix.RotationAxis(axis.normalize(), Math.acos(cosine));
  }

  _rotateReconstruction(rotation) {
    this._stopPointReveal();
    if (this.pointsMesh) {
      this.pointsMesh.bakeTransformIntoVertices(rotation);
      const positions = this.pointsMesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
      const points = [];
      for (let i = 0; i < positions.length; i += 3) points.push({ x: positions[i], y: positions[i + 1], z: positions[i + 2] });
      this.overviewPointBounds = this._robustBounds(points);
      this.sceneCenter = this.overviewPointBounds
        ? this.overviewPointBounds.min.add(this.overviewPointBounds.max).scale(0.5) : null;
    }
    // Replace records rather than mutating the original COLMAP parse cache.
    this.cameras = this.cameras.map(camera => ({ ...camera,
      center: BABYLON.Vector3.TransformCoordinates(camera.center, rotation),
      R: camera.R.multiply(rotation),
    }));
    this.photoGeneration++;
    for (const entry of this.photoTextures.values()) this._disposePhotoTexture(entry);
    this.photoTextures.clear();
    for (const mesh of this.cameraLineMeshes) mesh.dispose();
    this.cameraLineMeshes = [];
    this.frustumLines = [];
    this._createFrusta(this.cameras);
    this.frameScene();
  }

  alignZUp() {
    if (this.isBusy() || this.mode !== "scene") return;
    try {
      const rotation = this.alignmentRotation ? BABYLON.Matrix.Invert(this.alignmentRotation) : this._cameraPlaneRotation();
      this._rotateReconstruction(rotation);
      this.alignmentRotation = this.alignmentRotation ? null : rotation;
      this._updateAlignmentControl();
    } catch (error) {
      this._updateAlignmentControl(error.message);
    }
  }

  _updateAlignmentControl(message = "") {
    this.alignmentMessage = message;
    this._emitState();
  }

  _robustBounds(points) {
    const finite = points.filter(point => [point.x, point.y, point.z].every(Number.isFinite));
    if (!finite.length) return null;
    // Trim 5% from each end independently on each axis. Small sets (<20)
    // retain all coordinates, since they provide too little evidence to trim.
    const trim = Math.floor(finite.length * 0.05);
    const min = new BABYLON.Vector3(), max = new BABYLON.Vector3();
    for (const axis of ["x", "y", "z"]) {
      const values = finite.map(point => point[axis]).sort((a, b) => a - b);
      min[axis] = values[trim];
      max[axis] = values[values.length - 1 - trim];
    }
    return { min, max };
  }

  frameScene() {
    if (this.isBusy() && !this._pointStream) return;
    if (this._pointStream) this._pointStream.userNavigated = true;
    // Treat cameras separately so their small population is not overwhelmed
    // by the much larger point cloud, while rejecting stray cameras too.
    const bounds = [this.overviewPointBounds, this._robustBounds(this.cameras.map(camera => camera.center))].filter(Boolean);
    this._frameBounds(bounds);
  }

  viewFromAxis(axis) {
    // Reconstruction (X, Y, Z) maps to viewer (X, Z, Y).
    const directions = {
      x: [1, 0, 0], "-x": [-1, 0, 0],
      y: [0, 0, 1], "-y": [0, 0, -1],
      z: [0, 1, 0], "-z": [0, -1, 0],
    };
    if (!directions[axis]) return;
    if (this._pointStream) this._pointStream.userNavigated = true;
    const center = this.camera.target.clone();
    const radius = this.camera.radius;
    this.camera.stopInterpolation();
    this.camera.inertialAlphaOffset = this.camera.inertialBetaOffset = this.camera.inertialRadiusOffset = 0;
    this.camera.inertialPanningX = this.camera.inertialPanningY = 0;
    // Side views keep Z up; views along Z keep Y up to avoid an orbit pole.
    this.camera.upVector = new BABYLON.Vector3(...(axis.endsWith("z") ? [0, 0, 1] : [0, 1, 0]));
    this.cameraViewMode = "overview";
    this.camera.setTarget(center);
    this.camera.setPosition(center.add(new BABYLON.Vector3(...directions[axis]).scale(radius)));
    this.camera.rebuildAnglesAndRadius?.();
    this._updateOrthographicBounds();
    this._syncCameraVisuals();
  }

  _frameBounds(bounds) {
    if (!bounds.length) return;
    let min = bounds[0].min.clone(), max = bounds[0].max.clone();
    for (const bound of bounds.slice(1)) {
      min = BABYLON.Vector3.Minimize(min, bound.min);
      max = BABYLON.Vector3.Maximize(max, bound.max);
    }
    const center = min.add(max).scale(0.5);
    const radius = Math.max(0.1, max.subtract(min).length() * 1.2);
    this._setZoomScale(radius);
    this.camera.upVector = new BABYLON.Vector3(0, 1, 0);
    this.cameraViewMode = "overview";
    this.camera.fov = 0.8;
    this.camera.setTarget(center);
    this.camera.setPosition(center.add(new BABYLON.Vector3(0.4, 0.3, -1).normalize().scale(radius)));
    this.camera.minZ = radius / 1000;
    this.camera.rebuildAnglesAndRadius?.();
    this._syncCameraVisuals();
  }

  _autoFrame() {
    // look-at: sampled centroid if we have it
    const center = this.sceneCenter || new BABYLON.Vector3(0, 0, 0);
    this.camera.setTarget(center);

    if (this.cameras.length > 0) {
      this._gotoCam(0);
    } else if (this.pointsMesh) {
      // simple fit by radius if no cameras
      const bb = this.pointsMesh.getBoundingInfo().boundingBox;
      const size = Math.max(
        bb.maximumWorld.x - bb.minimumWorld.x,
        bb.maximumWorld.y - bb.minimumWorld.y,
        bb.maximumWorld.z - bb.minimumWorld.z
      );
      this._setZoomScale(size);
      this.camera.radius = size * 1.6;
    }
  }

  _gotoCam(i) {
    if (!this.cameras.length) return;
    this.camIndex = (i + this.cameras.length) % this.cameras.length;
    this.cameraViewMode = "photo";
    const selected = this.cameras[this.camIndex];
    const pos = selected.center;
    const forward = BABYLON.Vector3.TransformNormal(new BABYLON.Vector3(0, 0, 1), selected.R).normalize();
    const up = BABYLON.Vector3.TransformNormal(new BABYLON.Vector3(0, 1, 0), selected.R).normalize();
    const distance = Math.max(0.1, BABYLON.Vector3.Distance(pos, this.sceneCenter ?? BABYLON.Vector3.Zero()));
    this._setZoomScale(distance);
    this.camera.upVector = up;
    this.camera.setTarget(pos.add(forward.scale(distance)));
    this.camera.setPosition(pos);
    this.camera.rebuildAnglesAndRadius?.();
    this.camera.inertialAlphaOffset = this.camera.inertialBetaOffset = this.camera.inertialRadiusOffset = 0;
    this.camera.inertialPanningX = this.camera.inertialPanningY = 0;
    this._updateCameraFov();
    this.camera.minZ = Math.max(0.0001, distance / 1000);
    this.camera.maxZ = Math.max(10000, distance * 100);
    this._syncCameraVisuals();
    this._updateCurrentImageName();
    this._renderStats();
    this._emitPhotoState();
  }

  selectCamera(index) {
    if (!this.isBusy() && Number.isInteger(index)) this._gotoCam(index);
  }

  _updateCameraFov() {
    if (this.cameraViewMode === "overview") return;
    const k = this.cameras[this.camIndex]?.intrinsics;
    const aspect = this.engine.getAspectRatio(this.camera);
    this.camera.fov = k ? 2 * Math.atan(Math.max(k.height / (2 * k.fy), k.width / (2 * k.fx * aspect))) : 0.8;
  }

  nextCam() { this._gotoCam((this.camIndex ?? 0) + 1); }
  prevCam() { this._gotoCam((this.camIndex ?? 0) - 1); }

  _updateCurrentImageName() {
    if (!this.cameras.length) {
      this.currentImageName = "";
      return;
    }
    this.currentImageName = this.cameras[this.camIndex]?.name || "";
  }
}

// The inline bootstrap loads Babylon separately, then makes this factory available.
// React owns the canvas lifetime and every control surrounding it.
window.addEventListener("gtsfm-start-viewer", () => {
  window.createGtsfmViewer = (canvas) => new ColmapViewer(canvas);
  window.dispatchEvent(new Event("gtsfm-engine-ready"));
}, { once: true });
