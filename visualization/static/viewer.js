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
    this.camera.pinchPrecision = 120;
    this.camera.useNaturalPinchZoom = true;
    this.camera.lowerRadiusLimit = 0.1;
    this.camera.upperRadiusLimit = 10000;
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

    this.statsRoot = null;
    this.statsEls = null;
    this.pointsCount = 0;
    this.cameraCount = 0;
    this.currentImageName = "";
    this.statsVisible = true;
    this.loadingEls = null;
    this.hudRoot = null;
    this.resourceCache = new Map();
    this.resourceCacheLimit = 8;
    this.parsedPointsCache = new Map();
    this.parsedCamsCache = new Map();
    this.parsedCacheLimit = 4;
    this._busyState = null;
    this._busyListeners = new Set();
    this.overlayTheme = "dark";
    this.setOverlayTheme(this.overlayTheme);

    this.engine.runRenderLoop(() => this.scene.render());
    window.addEventListener("resize", () => this.engine.resize());
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
    this.camera.lowerRadiusLimit = effectiveVisibility ? 3.5 : 0.1;
    this.camera.upperRadiusLimit = effectiveVisibility ? 50 : 10000;
  }

  setGroundVisible(value) {
    this.showGround = !!value;
    this._setIdleGroundVisible(this.groundAvailable);
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
  }

  setPointSize(px) {
    this.ptSize = Math.max(0.5, Math.min(5, px));
    if (this.pointsMesh?.material) this.pointsMesh.material.pointSize = this.ptSize;
  }

  setGroundY(value) {
    const next = Number(value);
    if (!Number.isFinite(next)) return this.groundY;
    this.groundY = Math.max(-5, Math.min(5, next));
    if (this.idleGround) this.idleGround.position.y = this.groundY;
    return this.groundY;
  }

  setShowCams(v) {
    this.showCams = v;
    this.frustumGroup.setEnabled(!!v);
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

  setStatsElements(statsEls) {
    if (!statsEls) {
      this.statsRoot = null;
      this.statsEls = null;
      return;
    }
    const {
      root,
      cameraCount,
      pointCount,
      imageName,
      splatCount,
      sceneGroup,
      splatGroup,
    } = statsEls;
    this.statsRoot = root ?? null;
    this.statsEls = {
      cameraCount: cameraCount ?? null,
      pointCount: pointCount ?? null,
      imageName: imageName ?? null,
      splatCount: splatCount ?? null,
      sceneGroup: sceneGroup ?? null,
      splatGroup: splatGroup ?? null,
    };
    this._applyStatsVisibility();
    this._renderStats();
    this._setLoadingState({ active: false });
  }

  setHudElement(hudRoot) {
    this.hudRoot = hudRoot ?? null;
    this._applyHudMode();
  }

  setLoadingElements(loadingEls) {
    if (!loadingEls) {
      this.loadingEls = null;
      return;
    }
    const { overlay, bar, message } = loadingEls;
    this.loadingEls = {
      overlay: overlay ?? null,
      bar: bar ?? null,
      message: message ?? null,
    };
    this._setLoadingState({ active: false, progress: 0 });
  }

  isBusy() {
    return !!this._busyState;
  }

  onBusyChange(listener) {
    if (typeof listener !== "function") return () => {};
    this._busyListeners.add(listener);
    try {
      listener(this.isBusy(), this._busyState?.reason ?? null);
    } catch (err) {
      console.warn("Busy listener threw on registration:", err);
    }
    return () => this._busyListeners.delete(listener);
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

  _notifyBusyChange() {
    const busy = this.isBusy();
    const reason = this._busyState?.reason ?? null;
    for (const listener of this._busyListeners) {
      try {
        listener(busy, reason);
      } catch (err) {
        console.warn("Busy listener threw:", err);
      }
    }
  }

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

  _applyStatsVisibility() {
    if (!this.statsRoot) return;
    this.statsRoot.style.display = this.statsVisible ? "" : "none";
  }

  _applyStatsMode() {
    if (!this.statsEls) return;
    const { sceneGroup, splatGroup } = this.statsEls;
    const showScene = this.mode !== "splat";
    const showSplats = this.mode === "splat";
    if (sceneGroup) sceneGroup.style.display = showScene ? "grid" : "none";
    if (splatGroup) splatGroup.style.display = showSplats ? "grid" : "none";
  }

  _applyHudMode() {
    if (!this.hudRoot) return;
    this.hudRoot.classList.remove("hud-hidden");
    this.hudRoot.classList.toggle("hud-splat-mode", this.mode === "splat");
  }

  _renderStats() {
    if (!this.statsEls) return;
    this._applyStatsVisibility();
    this._applyStatsMode();

    const fmt = (value) => (typeof value === "number" && !Number.isNaN(value) ? value.toLocaleString() : "—");
    const { cameraCount, pointCount, imageName, splatCount } = this.statsEls;

    if (this.mode === "splat") {
      if (splatCount) {
        const text = this.splatsPointCount ? fmt(this.splatsPointCount) : "0";
        splatCount.textContent = text;
      }
      return;
    }

    if (pointCount) {
      const text = this.pointsCount ? fmt(this.pointsCount) : "0";
      pointCount.textContent = text;
    }
    if (cameraCount) {
      const text = this.cameraCount ? fmt(this.cameraCount) : "0";
      cameraCount.textContent = text;
    }
    if (imageName) {
      imageName.textContent = this.currentImageName || "—";
    }
    if (splatCount) {
      const text = this.splatsPointCount ? fmt(this.splatsPointCount) : "0";
      splatCount.textContent = text;
    }
  }

  _setLoadingState({ active = null, message = null, progress = null } = {}) {
    if (!this.loadingEls) return;
    const { overlay, bar, message: msgEl } = this.loadingEls;
    if (overlay && active !== null) {
      if (active) {
        overlay.classList.add("active");
      } else {
        overlay.classList.remove("active");
      }
    }
    if (msgEl && message !== null) {
      msgEl.textContent = message;
    }
    if (bar && progress !== null && Number.isFinite(progress)) {
      const clamped = Math.max(0, Math.min(1, progress));
      bar.style.width = `${clamped * 100}%`;
      bar.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
    }
  }

  _updateLoadingProgress(progress) {
    this._setLoadingState({ progress });
  }

  async _yieldFrame() {
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }

  async loadScene({ pointsUrl, imagesUrl, camerasUrl = null, scenePath = null, splatsUrl = null }) {
    const busyToken = this._beginBusy("scene");
    try {
      await this._clear();
      this._setIdleGroundVisible(true);
      this.mode = "scene";
      this.splatsUrl = splatsUrl;
      this._applyHudMode();

      this._setLoadingState({ active: true, message: "Loading reconstruction…", progress: 0 });

      const fetchOpts = { cache: "no-store" };
      const cachedPoints = pointsUrl ? this.parsedPointsCache.get(pointsUrl) : null;
      const cachedCameras = imagesUrl ? this.parsedCamsCache.get(imagesUrl) : null;
      const needPoints = !cachedPoints;
      const needCameras = imagesUrl ? !cachedCameras : false;

      const [pointsText, imagesText, intrinsicsText] = await Promise.all([
        needPoints ? this._fetchResource(pointsUrl, "text", fetchOpts) : Promise.resolve(null),
        needCameras ? this._fetchResource(imagesUrl, "text", fetchOpts) : Promise.resolve(null),
        camerasUrl ? this._fetchResource(camerasUrl, "text").catch(() => "") : Promise.resolve(""),
      ]);

      if (needPoints) {
        this._setLoadingState({ message: "Parsing points…", progress: 0.35 });
      }

      const points = cachedPoints ?? this._parsePoints(pointsText ?? "");
      if (needPoints && pointsUrl) {
        this._rememberParsed(this.parsedPointsCache, pointsUrl, points, this.parsedCacheLimit);
      }

      if (imagesUrl) {
        this.cameras = cachedCameras ?? (imagesText ? this._parseCams(imagesText) : []);
        if (needCameras) {
          this._rememberParsed(this.parsedCamsCache, imagesUrl, this.cameras, this.parsedCacheLimit);
        }
      } else {
        this.cameras = [];
      }
      const intrinsics = this._parseIntrinsics(intrinsicsText);
      // Do not mutate cached camera records when switching reconstructions.
      this.cameras = this.cameras.map((camera) => ({ ...camera, intrinsics: intrinsics.get(camera.cameraId) }));

      this._setLoadingState({ message: "Building point cloud…", progress: 0.6 });

      if (points.length) await this._createPCS(points);
      if (this.cameras.length) this._createFrusta(this.cameras);

      this.pointsCount = points.length;
      this.cameraCount = this.cameras.length;

      // centroid from a 100-point sample (fallback to bbox center if needed)
      this.sceneCenter = points.length
        ? this._centroidSample(points, 100)
        : (this.pointsMesh
          ? this.pointsMesh.getBoundingInfo().boundingBox.centerWorld.clone()
          : new BABYLON.Vector3(0, 0, 0));

      // remember which camera we're on for next/prev UI
      this.camIndex = 0;
      this._updateCurrentImageName();
      this._renderStats();

      this._autoFrame();
      void this._loadPhotos(scenePath);
      this._setLoadingState({ progress: 1 });
    } finally {
      this._setLoadingState({ active: false });
      this._endBusy(busyToken);
    }
  }

  async loadSplatsFile({ splatsUrl, label = "" }) {
    const busyToken = this._beginBusy("splat");
    try {
      await this._clear();
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

      const bb = this.splatsMesh.getBoundingInfo().boundingBox;
      const center = bb.centerWorld.clone();
      this.sceneCenter = center;
      this.camera.setTarget(center);
      const size = Math.max(
        bb.maximumWorld.x - bb.minimumWorld.x,
        bb.maximumWorld.y - bb.minimumWorld.y,
        bb.maximumWorld.z - bb.minimumWorld.z
      ) || 1.0;
      this.camera.radius = size * 1.6;
      this.camera.rebuildAnglesAndRadius?.();
      return true;
    } finally {
      this._endBusy(busyToken);
    }
  }

  async _clear() {
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
    this.splatsPointCount = 0;
    this.sceneCenter = null;

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
        pts.push({
          x: parseFloat(s[1]), y: parseFloat(s[3]), z: parseFloat(s[2]),
          r: parseInt(s[4]) / 255, g: parseInt(s[5]) / 255, b: parseInt(s[6]) / 255
        });
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

    updateLoading({ active: true, message: "Loading splat file…", progress: 0 });

    const promise = (async () => {
      const buffer = await this._fetchResource(this.splatsUrl, "arrayBuffer", {
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
    this.splatsMesh = mesh;
    this._splatsMeshToken = token ?? null;
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
    const { cacheResult = true, ...requestInit } = fetchOpts;
    const key = `${kind}:${url}`;
    if (cacheResult && this.resourceCache.has(key)) {
      const cached = this.resourceCache.get(key);
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
    const text = await response.text();
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

  _emitPhotoState() {
    window.dispatchEvent(new CustomEvent("gtsfm-photo-state", { detail: this.getPhotoState() }));
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

  frameScene() {
    if (this.isBusy() || !this.cameras.length) return;
    const bounds = this.pointsMesh?.getBoundingInfo().boundingBox;
    let min = bounds?.minimumWorld.clone() ?? this.cameras[0].center.clone();
    let max = bounds?.maximumWorld.clone() ?? min.clone();
    for (const camera of this.cameras) {
      min = BABYLON.Vector3.Minimize(min, camera.center);
      max = BABYLON.Vector3.Maximize(max, camera.center);
    }
    const center = min.add(max).scale(0.5);
    const radius = Math.max(0.1, max.subtract(min).length() * 1.2);
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

// ------------------- app bootstrap -------------------

function resultsForRun(items, baseDir, runRoot) {
  const normalize = (path) => path.replaceAll("\\", "/").replace(/\/+$/, "");
  const base = normalize(baseDir);
  const root = normalize(runRoot);
  if (root !== base && !root.startsWith(`${base}/`)) return [];
  const relativeRoot = root === base ? "" : root.slice(base.length + 1);
  return items.filter((item) => !relativeRoot || item.rel_path === relativeRoot || item.rel_path.startsWith(`${relativeRoot}/`));
}

function preferredRunResult(items) {
  // Prefer the final root reconstruction over per-cluster results and BA diagnostics.
  const sceneFolders = ["merged_retriangulated", "merged", "ba_output", "vggt", "anysplat"];
  const rank = (item) => {
    if (item.kind === "splat") return 0;
    const parts = item.rel_path.split("/");
    const stage = sceneFolders.indexOf(parts.at(-1));
    return stage < 0 ? Infinity : 1 + parts.length * sceneFolders.length + stage;
  };
  return items.filter((item) => Number.isFinite(rank(item))).sort((a, b) => rank(a) - rank(b))[0];
}

async function boot() {
  const info = document.getElementById("info");
  const infoCountEl = info?.querySelector('[data-role="count"]') ?? null;
  const infoTitleEl = info?.querySelector('[data-role="title"]') ?? null;
  const infoPathEl = info?.querySelector('[data-role="path"]') ?? null;
  const listEl = document.getElementById("sceneList");
  const filterEl = document.getElementById("filter");
  const canvas = document.getElementById("renderCanvas");
  const viewer = new ColmapViewer(canvas);
  window.gtsfmViewer = viewer;
  window.dispatchEvent(new CustomEvent("gtsfm-viewer-ready", { detail: viewer }));
  const statsRoot = document.getElementById("sceneStats");
  viewer.setStatsElements({
    root: statsRoot,
    cameraCount: document.getElementById("statCameras"),
    pointCount: document.getElementById("statPoints"),
    imageName: document.getElementById("statImageName"),
    splatCount: document.getElementById("statSplats"),
    sceneGroup: statsRoot?.querySelector('[data-mode="scene"]') ?? null,
    splatGroup: statsRoot?.querySelector('[data-mode="splat"]') ?? null,
  });
  viewer.setLoadingElements({
    overlay: document.getElementById("loadingOverlay"),
    bar: document.getElementById("loadingProgress"),
    message: document.getElementById("loadingMessage"),
  });
  viewer.setHudElement(document.getElementById("hud"));

  const updateInteractionLock = (busy) => {
    listEl.classList.toggle("is-disabled", !!busy);
    if (filterEl) filterEl.disabled = !!busy;
  };
  viewer.onBusyChange((busy) => updateInteractionLock(busy));
  updateInteractionLock(viewer.isBusy());

  const setInfoState = ({ count, title, path, isError = false }) => {
    if (!info) return;
    info.classList.toggle("info-error", !!isError);
    if (infoCountEl && typeof count !== "undefined" && count !== null) {
      infoCountEl.textContent = `${count}`;
    }
    if (infoTitleEl && typeof title !== "undefined" && title !== null) {
      infoTitleEl.textContent = title;
    }
    if (infoPathEl && typeof path !== "undefined" && path !== null) {
      infoPathEl.textContent = path;
      infoPathEl.title = path;
    }
  };

  setInfoState({
    count: "…",
    title: "Loading reconstructions",
    path: "Fetching data from server…",
    isError: false,
  });

  let allItems = [];
  const collapsedDirectories = new Set();
  const searchParams = new URLSearchParams(window.location.search);
  const requestedRun = searchParams.get("run");

  try {
    const data = await fetch("/api/scenes").then(r => r.ok ? r.json() : Promise.reject(`Fetch failed: ${r.status} ${r.statusText}`.trim()));
    const items = requestedRun ? resultsForRun(data.items, data.base_dir, requestedRun) : data.items;
    const count = items.length;
    const countLabel = count === 0 ? "0" : count.toLocaleString();
    const noun = count === 1 ? "item" : "items";
    const title = count === 0 ? (requestedRun ? "No results found for this run" : "No items found") : `${countLabel} ${noun} ${requestedRun ? "in this run" : "ready to explore"}`;
    const baseDir = data.base_dir || "—";
    setInfoState({
      count: countLabel,
      title,
      path: requestedRun ? `Run directory: ${requestedRun}` : `Base directory: ${baseDir}`,
      isError: Boolean(requestedRun && count === 0),
    });
    allItems = items.map((item, index) => ({ ...item, originalIndex: index }));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setInfoState({
      count: "!",
      title: "Error loading reconstruction data",
      path: message || "Please check the server logs.",
      isError: true,
    });
    console.error(error);
    return;
  }

  const buildTree = (items) => {
    const tree = {};
    items.forEach(item => {
      const parts = item.rel_path.split('/');
      let cur = tree;
      parts.forEach((part, i) => {
        if (!cur[part]) cur[part] = {};
        if (i === parts.length - 1) {
          cur[part].__isLeaf = true;
          cur[part].__item = item;
        }
        cur = cur[part];
      });
    });
    return tree;
  };

  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const renderTree = (node, parentParts = [], forceExpanded = false) => {
    let html = '<ul>';
    const sortedKeys = Object.keys(node).sort();
    for (const key of sortedKeys) {
      if (key.startsWith('__')) continue;
      const child = node[key];
      if (child.__isLeaf) {
        const item = child.__item;
        const finalFolderName = item.rel_path.split('/').pop() || item.label;
        const exportControls = item.splat_rel_path ? `
              <div class="result-splat-download" data-export-path="${encodeURIComponent(item.splat_rel_path)}">
                <span class="splat-format-label">.PLY</span>
                <a href="/api/splats/export?path=${encodeURIComponent(item.splat_rel_path)}&format=ply" download>Download</a>
              </div>` : "";
        html += `
          <li>
            <div class="item" data-idx="${item.originalIndex}" data-kind="${item.kind || "scene"}">
              <div>${escapeHtml(finalFolderName)}</div>
              <small>${escapeHtml(item.rel_path)}</small>
              ${exportControls}
            </div>
          </li>`;
      } else {
        const directoryParts = [...parentParts, key];
        const directoryPath = directoryParts.join("/");
        const isCollapsed = !forceExpanded && collapsedDirectories.has(directoryPath);
        html += `
          <li class="directory-branch${isCollapsed ? " is-collapsed" : ""}">
            <button class="directory-node" type="button" data-directory="${encodeURIComponent(directoryPath)}" aria-expanded="${!isCollapsed}">
              <span class="directory-chevron" aria-hidden="true"></span>
              <span>${escapeHtml(key)}</span>
            </button>
            ${renderTree(child, directoryParts, forceExpanded)}
          </li>`;
      }
    }
    return html + '</ul>';
  };

  const renderList = (query = "") => {
    const needle = query.trim().toLowerCase();
    const filtered = allItems.filter(it => it.label.toLowerCase().includes(needle));
    if (filtered.length === 0) {
      listEl.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b;">No items found.</div>';
      return;
    }
    const sceneTree = buildTree(filtered);
    listEl.innerHTML = renderTree(sceneTree, [], Boolean(needle));
    listEl.querySelectorAll(".directory-node").forEach(button => {
      button.addEventListener("click", () => {
        const directoryPath = decodeURIComponent(button.dataset.directory || "");
        const branch = button.closest(".directory-branch");
        const shouldCollapse = !branch.classList.contains("is-collapsed");
        branch.classList.toggle("is-collapsed", shouldCollapse);
        button.setAttribute("aria-expanded", String(!shouldCollapse));
        if (shouldCollapse) collapsedDirectories.add(directoryPath);
        else collapsedDirectories.delete(directoryPath);
      });
    });
    listEl.querySelectorAll(".item").forEach(el => {
      const exportControls = el.querySelector(".result-splat-download");
      if (exportControls) {
        exportControls.addEventListener("click", (event) => event.stopPropagation());
      }
      el.onclick = async () => {
        if (viewer.isBusy()) return;
        listEl.querySelectorAll(".item").forEach(n => n.classList.remove("active"));
        el.classList.add("active");
        const originalIndex = parseInt(el.dataset.idx, 10);
        const item = allItems.find(it => it.originalIndex === originalIndex);
        if (!item) return;
        console.log("Loading scene:", item);
        if ((item.kind || "scene") === "splat") {
          await viewer.loadSplatsFile({ splatsUrl: item.splats, label: item.label });
        } else {
          await viewer.loadScene({ pointsUrl: item.points, imagesUrl: item.images, camerasUrl: item.cameras, scenePath: item.rel_path, splatsUrl: item.splats });
        }
      };
    });
  };


  // Default search term
  filterEl.value = "";
  renderList(filterEl.value);
  const initialView = searchParams.get("view");
  if (allItems.length > 0 && !filterEl.value && initialView === "results") {
    const candidates = Array.from(listEl.querySelectorAll(".item"));
    if (requestedRun) {
      const result = preferredRunResult(allItems);
      if (result) {
        candidates.find((el) => Number(el.dataset.idx) === result.originalIndex)?.click();
      } else {
        setInfoState({ title: "No final reconstruction found for this run", isError: true });
      }
    } else {
      const preferred = candidates.find((el) => (el.dataset.kind ?? "scene") !== "splat");
      (preferred ?? candidates[0])?.click();
    }
  }
  filterEl.addEventListener("input", () => renderList(filterEl.value));

  document.getElementById("toggleCams").addEventListener("change", (e) => viewer.setShowCams(e.target.checked));
  document.getElementById("ptSize").addEventListener("input", (e) => viewer.setPointSize(parseInt(e.target.value, 10)));

  const groundY = document.getElementById("groundY");
  const groundYValue = document.getElementById("groundYValue");
  groundY.addEventListener("input", (event) => {
    const value = viewer.setGroundY(event.target.value);
    groundYValue.value = value.toFixed(1);
  });

  const groundToggle = document.getElementById("toggleGround");
  if (groundToggle) {
    const syncGroundButton = () => {
      const visible = viewer.isGroundVisible();
      groundToggle.textContent = visible ? "Hide plane" : "Show plane";
      groundToggle.classList.toggle("is-active", visible);
      groundToggle.setAttribute("aria-pressed", String(visible));
    };
    groundToggle.addEventListener("click", () => {
      viewer.toggleGroundVisible();
      syncGroundButton();
    });
    syncGroundButton();
  }

  const backgroundSelect = document.getElementById("backgroundSelect");
  const backgroundColors = {
    dark: new BABYLON.Color4(0.055, 0.063, 0.071, 1.0),
    graphite: new BABYLON.Color4(0.10, 0.11, 0.12, 1.0),
    "light-gray": new BABYLON.Color4(0.85, 0.85, 0.83, 1.0),
    white: new BABYLON.Color4(0.98, 0.98, 0.96, 1.0),
  };
  backgroundSelect.addEventListener("change", (event) => {
    viewer.scene.clearColor = backgroundColors[event.target.value] ?? backgroundColors.dark;
    viewer.setOverlayTheme(event.target.value);
  });

  const prevBtn = document.getElementById("prevCamBtn");
  document.getElementById("overviewBtn")?.addEventListener("click", () => viewer.frameScene());
  const nextBtn = document.getElementById("nextCamBtn");
  prevBtn?.addEventListener("click", () => viewer.prevCam());
  nextBtn?.addEventListener("click", () => viewer.nextCam());

  const statsBtn = document.getElementById("toggleStats");
  if (statsBtn) {
    const syncStatsButtonLabel = () => {
      statsBtn.textContent = viewer.isStatsVisible() ? "Hide Stats" : "Show Stats";
    };
    statsBtn.addEventListener("click", () => {
      viewer.toggleStatsVisible();
      syncStatsButtonLabel();
    });
    syncStatsButtonLabel();
  }
}

window.addEventListener("DOMContentLoaded", boot);
