// visualization/static/viewer.js

class ColmapViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.85, 0.85, 0.85, 1.0);

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

    // Z-up for interaction
    this.camera.upVector = new BABYLON.Vector3(0, 0, -1);


    // Inputs & limits tuned to avoid “locked” feeling
    this.camera.angularSensibilityX = 1000;
    this.camera.angularSensibilityY = 1000;
    this.camera.panningSensibility = 1000;
    this.camera.wheelPrecision = 40;
    this.camera.lowerRadiusLimit = 0.1;
    this.camera.upperRadiusLimit = 10000;
    this.camera.lowerBetaLimit = 0.001;
    this.camera.upperBetaLimit = Math.TWO_PI - 0.001;

    this.light = new BABYLON.HemisphericLight("H", new BABYLON.Vector3(0, 1, 0), this.scene);
    this.light.intensity = 0.85;

    this.pcs = null;
    this.pointsMesh = null;
    this.frustumGroup = new BABYLON.TransformNode("frusta", this.scene);
    this.cameras = [];

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

    this.engine.runRenderLoop(() => this.scene.render());
    window.addEventListener("resize", () => this.engine.resize());
  }

  setPointSize(px) {
    this.ptSize = Math.max(0.5, Math.min(5, px));
    if (this.pointsMesh?.material) this.pointsMesh.material.pointSize = this.ptSize;
  }

  setShowCams(v) {
    this.showCams = v;
    this.frustumGroup.setEnabled(!!v);
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
    const hideHud = this.mode === "splat";
    if (hideHud) {
      this.hudRoot.classList.add("hud-hidden");
    } else {
      this.hudRoot.classList.remove("hud-hidden");
    }
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

  async loadScene({ pointsUrl, imagesUrl, splatsUrl = null }) {
    const busyToken = this._beginBusy("scene");
    try {
      await this._clear();
      this.mode = "scene";
      this.splatsUrl = splatsUrl;
      this._applyHudMode();

      this._setLoadingState({ active: true, message: "Loading reconstruction…", progress: 0 });

      const fetchOpts = { cache: "no-store" };
      const cachedPoints = pointsUrl ? this.parsedPointsCache.get(pointsUrl) : null;
      const cachedCameras = imagesUrl ? this.parsedCamsCache.get(imagesUrl) : null;
      const needPoints = !cachedPoints;
      const needCameras = imagesUrl ? !cachedCameras : false;

      const [pointsText, imagesText] = await Promise.all([
        needPoints ? this._fetchResource(pointsUrl, "text", fetchOpts) : Promise.resolve(null),
        needCameras ? this._fetchResource(imagesUrl, "text", fetchOpts) : Promise.resolve(null),
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
      console.log('first camera center:', this.cameras[0]?.center);
      console.log('scene extent:', this._sceneExtent());
      console.log(`Loaded: #points=${points.length}, #cams=${this.cameras.length}`);

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
        return;
      }

      if (this._activeSplatsToken !== token) {
        return;
      }

      if (!this.splatsMesh) {
        this._renderStats();
        return;
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
    } finally {
      this._endBusy(busyToken);
    }
  }

  async _clear() {
    this.mode = "idle";
    this.cameras = [];
    this.pointsCount = 0;
    this.cameraCount = 0;
    this.currentImageName = "";
    this._applyHudMode();

    if (this.pointsMesh) { this.pointsMesh.dispose(); this.pointsMesh = null; }
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
        // flip Y for Babylon’s Y-up RH frame
        pts.push({
          x: parseFloat(s[1]), y: -parseFloat(s[2]), z: parseFloat(s[3]),
          r: parseInt(s[4]) / 255, g: parseInt(s[5]) / 255, b: parseInt(s[6]) / 255
        });
      }
    }
    return pts;
  }

  _parseCams(text) {
    const cams = [];
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

        // Flip into viewer coords (Y flip)
        const flip = BABYLON.Matrix.Scaling(1, -1, 1);
        const center = new BABYLON.Vector3(centerWorld.x, -centerWorld.y, centerWorld.z);
        const Rcam = flip.multiply(Rcw).multiply(flip);

        const name = s.length > 9 ? s.slice(9).join(" ") : "";
        cams.push({ center, R: Rcam, name });

        // Skip 2D measurements line if present
        if (i + 1 < lines.length && /^\d/.test((lines[i + 1] || "").trim())) i++;
      }
    }
    return cams;
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
      try {
        this.splatsMesh.thinInstanceSetBuffer("matrix", null);
        this.splatsMesh.thinInstanceSetBuffer("color", null);
      } catch (err) {
        console.warn("Unable to clear splat thin instances:", err);
      }
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
      const buffer = await this._fetchResource(this.splatsUrl, "arrayBuffer", { cache: "no-store" });
      if (!isTokenActive()) return;

      const splats = this._parseSplatsPly(buffer);
      if (!splats.length) {
        if (isTokenActive()) {
          throw new Error("Parsed gaussian splats contained no points.");
        }
        return;
      }

      updateLoading({ message: "Rendering splats…", progress: 0.15 });
      await this._createSplatsMesh(splats, token);

      if (!isTokenActive()) return;

      this.splatsPointCount = splats.length;
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

  async _createSplatsMesh(splats, token = null) {
    const isTokenActive = () => !token || (this._activeSplatsToken === token && this.mode === "splat");
    if (!isTokenActive()) return;

    const base = BABYLON.MeshBuilder.CreateSphere(
      "splatBase",
      { diameter: 2, segments: 6 },
      this.scene
    );
    base.isVisible = true;
    base.isPickable = false;
    base.alwaysSelectAsActiveMesh = true;
    base.thinInstanceEnablePicking = false;

    const material = new BABYLON.StandardMaterial("splatsMaterial", this.scene);
    material.disableLighting = true;
    material.emissiveColor = new BABYLON.Color3(1, 1, 1);
    material.backFaceCulling = false;
    material.alphaMode = BABYLON.Constants.ALPHA_COMBINE;
    material.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;
    material.zWrite = false;
    material.useInstancedBuffers = true;
    if ("useInstancingColor" in material) {
      material.useInstancingColor = true;
    }
    base.material = material;

    const bailIfStale = () => {
      if (isTokenActive()) return false;
      material.dispose();
      base.dispose();
      return true;
    };

    const matrixBuffer = new Float32Array(splats.length * 16);
    const colorBuffer = new Float32Array(splats.length * 4);
    const tmpScale = new BABYLON.Vector3();
    const tmpQuat = new BABYLON.Quaternion();
    const tmpPos = new BABYLON.Vector3();
    const tmpMatrix = new BABYLON.Matrix();
    const MIN_SCALE = 0.01;

    for (let i = 0; i < splats.length; i++) {
      const s = splats[i];
      const scale = s.scale ?? [0, 0, 0];
      const rot = s.rotation ?? [0, 0, 0, 1];
      const opacity = Number.isFinite(s.opacity) ? Math.min(1, Math.max(0, s.opacity)) : 1.0;

      const sx = Number.isFinite(scale[0]) ? Math.max(MIN_SCALE, Math.exp(scale[0])) : 1;
      const sy = Number.isFinite(scale[1]) ? Math.max(MIN_SCALE, Math.exp(scale[1])) : 1;
      const sz = Number.isFinite(scale[2]) ? Math.max(MIN_SCALE, Math.exp(scale[2])) : 1;
      tmpScale.set(sx, sy, sz);

      const qx = Number.isFinite(rot[0]) ? rot[0] : 0;
      const qy = Number.isFinite(rot[1]) ? rot[1] : 0;
      const qz = Number.isFinite(rot[2]) ? rot[2] : 0;
      const qw = Number.isFinite(rot[3]) ? rot[3] : 1;
      tmpQuat.set(qx, qy, qz, qw);
      tmpQuat.normalize();
      tmpPos.set(s.x, s.y, s.z);

      BABYLON.Matrix.ComposeToRef(tmpScale, tmpQuat, tmpPos, tmpMatrix);
      tmpMatrix.copyToArray(matrixBuffer, i * 16);

      colorBuffer[i * 4 + 0] = s.r;
      colorBuffer[i * 4 + 1] = s.g;
      colorBuffer[i * 4 + 2] = s.b;
      colorBuffer[i * 4 + 3] = opacity;

      if (splats.length > 1000 && i % 2000 === 0) {
        if (bailIfStale()) return;
        if (isTokenActive()) this._updateLoadingProgress(i / splats.length);
        await this._yieldFrame();
        if (bailIfStale()) return;
      }
    }
    if (bailIfStale()) return;
    if (isTokenActive()) this._updateLoadingProgress(0.98);

    base.thinInstanceSetBuffer("matrix", matrixBuffer, 16);
    base.thinInstanceSetBuffer("color", colorBuffer, 4);
    base.thinInstanceRefreshBoundingInfo();
    base.setEnabled(true);
    if (bailIfStale()) return;
    if (isTokenActive()) this._updateLoadingProgress(1.0);
    if (splats.length > 1000) {
      await this._yieldFrame();
      if (bailIfStale()) return;
    }
    this.splatsMesh = base;
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
      const opacity = Number.isFinite(opacityRaw) ? Math.min(1, Math.max(0, opacityRaw)) : 1.0;
      const scale0 = idx.scale0 >= 0 ? parseFloat(parts[idx.scale0]) : 0;
      const scale1 = idx.scale1 >= 0 ? parseFloat(parts[idx.scale1]) : 0;
      const scale2 = idx.scale2 >= 0 ? parseFloat(parts[idx.scale2]) : 0;
      const rot0 = idx.rot0 >= 0 ? parseFloat(parts[idx.rot0]) : 0;
      const rot1 = idx.rot1 >= 0 ? parseFloat(parts[idx.rot1]) : 0;
      const rot2 = idx.rot2 >= 0 ? parseFloat(parts[idx.rot2]) : 0;
      const rot3 = idx.rot3 >= 0 ? parseFloat(parts[idx.rot3]) : 1;

      points.push({
        x,
        y: -y,
        z,
        r,
        g,
        b,
        opacity,
        scale: [scale0, scale1, scale2],
        rotation: [rot0, rot1, rot2, rot3],
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
      const alpha = Number.isFinite(opacity) ? Math.min(1, Math.max(0, opacity)) : 1.0;
      points.push({
        x,
        y: -y,
        z,
        r,
        g,
        b,
        opacity: alpha,
        scale: [scale0, scale1, scale2],
        rotation: [rot0, rot1, rot2, rot3 || 1],
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
    const key = `${kind}:${url}`;
    if (this.resourceCache.has(key)) {
      const cached = this.resourceCache.get(key);
      if (kind === "arrayBuffer") {
        return cached.slice(0);
      }
      return cached;
    }
    const response = await fetch(url, fetchOpts);
    if (!response.ok) {
      throw new Error(`Failed to load ${url}`);
    }
    if (kind === "arrayBuffer") {
      const buffer = await response.arrayBuffer();
      this._rememberResource(key, buffer);
      return buffer.slice(0);
    }
    const text = await response.text();
    this._rememberResource(key, text);
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

  _createFrusta(cams) {
    const size = this._sceneExtent();
    const frustumScale = 0.01 * size;   // 1% of scene extent
    const pivotDiam = 0.002 * size;   // 0.2% sphere

    for (const c of cams) {
      const node = new BABYLON.TransformNode("camNode", this.scene);
      node.position.copyFrom(c.center);
      node.rotationQuaternion = BABYLON.Quaternion.FromRotationMatrix(c.R);
      node.parent = this.frustumGroup;

      const frustum = this._frustumLinesLocal(frustumScale);
      frustum.parent = node;
      frustum.renderingGroupId = 1; // draw after points
      // If still hard to see, try:
      frustum.alwaysSelectAsActiveMesh = true;
      const m = new BABYLON.StandardMaterial("fmat", this.scene);
      m.emissiveColor = new BABYLON.Color3(0, 0.5, 0);
      frustum.material = m;

      const pivot = BABYLON.MeshBuilder.CreateSphere("camPivot", { diameter: pivotDiam }, this.scene);
      const pivotMat = new BABYLON.StandardMaterial("camPivotMat", this.scene);
      pivotMat.emissiveColor = new BABYLON.Color3(0, 0, 0.5);
      pivotMat.disableLighting = true;
      pivot.material = pivotMat;
      pivot.isPickable = false;
      pivot.parent = node;
    }
    this.setShowCams(this.showCams);
  }

  _frustumLinesLocal(scale) {
    const origin = BABYLON.Vector3.Zero();
    const corners = [
      new BABYLON.Vector3(-0.4, -0.3, 1),
      new BABYLON.Vector3(0.4, -0.3, 1),
      new BABYLON.Vector3(0.4, 0.3, 1),
      new BABYLON.Vector3(-0.4, 0.3, 1),
    ].map(v => v.scale(scale));

    const lines = [
      [origin, corners[0]], [origin, corners[1]], [origin, corners[2]], [origin, corners[3]],
      [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],
    ];

    const frustum = BABYLON.MeshBuilder.CreateLineSystem("frustumLines", { lines }, this.scene);
    frustum.color = new BABYLON.Color3(1, 0.45, 0.25);
    frustum.isPickable = false;
    frustum.renderingGroupId = 1; // draw after points (optional)
    return frustum;
  }

  // ------------------- camera framing -------------------

  _autoFrame() {
    // look-at: sampled centroid if we have it
    const center = this.sceneCenter || new BABYLON.Vector3(0, 0, 0);
    this.camera.setTarget(center);

    if (this.cameras.length > 0) {
      const pos = this.cameras[this.camIndex].center;
      this.camera.setPosition(pos);
      // recompute alpha/beta/radius from pos/target so orbit feels natural
      this.camera.rebuildAnglesAndRadius?.();
      this._updateCurrentImageName();
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
    const center = this.sceneCenter || new BABYLON.Vector3(0, 0, 0);
    const pos = this.cameras[this.camIndex].center;
    this.camera.setTarget(center);
    this.camera.setPosition(pos);
    this.camera.rebuildAnglesAndRadius?.();
    const R = this.cameras[this.camIndex].R;
    const upWorld = BABYLON.Vector3.TransformCoordinates(new BABYLON.Vector3(0, 1, 0), R);
    this.camera.upVector = upWorld;
    this._updateCurrentImageName();
    this._renderStats();
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

async function boot() {
  const info = document.getElementById("info");
  const infoCountEl = info?.querySelector('[data-role="count"]') ?? null;
  const infoTitleEl = info?.querySelector('[data-role="title"]') ?? null;
  const infoPathEl = info?.querySelector('[data-role="path"]') ?? null;
  const listEl = document.getElementById("sceneList");
  const filterEl = document.getElementById("filter");
  const canvas = document.getElementById("renderCanvas");
  const viewer = new ColmapViewer(canvas);
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

  try {
    const data = await fetch("/api/scenes").then(r => r.ok ? r.json() : Promise.reject(`Fetch failed: ${r.statusText}`));
    const count = Number.isFinite(data.count) ? data.count : 0;
    const countLabel = count === 0 ? "0" : count.toLocaleString();
    const noun = count === 1 ? "item" : "items";
    const title = count === 0 ? "No items found" : `${countLabel} ${noun} ready to explore`;
    const baseDir = data.base_dir || "—";
    setInfoState({
      count: countLabel,
      title,
      path: `Base directory: ${baseDir}`,
      isError: false,
    });
    allItems = data.items.map((item, index) => ({ ...item, originalIndex: index }));
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

  const renderTree = (node) => {
    let html = '<ul>';
    const sortedKeys = Object.keys(node).sort();
    for (const key of sortedKeys) {
      if (key.startsWith('__')) continue;
      const child = node[key];
      if (child.__isLeaf) {
        const item = child.__item;
        const finalFolderName = item.rel_path.split('/').pop() || item.label;
        html += `
          <li>
            <div class="item" data-idx="${item.originalIndex}" data-kind="${item.kind || "scene"}">
              <div>${finalFolderName}</div>
              <small>${item.rel_path}</small>
            </div>
          </li>`;
      } else {
        html += `<li><div class="directory-node">${key}</div>${renderTree(child)}</li>`;
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
    listEl.innerHTML = renderTree(sceneTree);
    listEl.querySelectorAll(".item").forEach(el => {
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
          await viewer.loadScene({ pointsUrl: item.points, imagesUrl: item.images, splatsUrl: item.splats });
        }
      };
    });
  };


  // Default search term
  filterEl.value = "";
  renderList(filterEl.value);
  if (allItems.length > 0 && !filterEl.value) {
    const first = listEl.querySelector(".item");
    if (first) first.click();
  }
  filterEl.addEventListener("input", () => renderList(filterEl.value));

  document.getElementById("toggleCams").addEventListener("change", (e) => viewer.setShowCams(e.target.checked));
  document.getElementById("ptSize").addEventListener("input", (e) => viewer.setPointSize(parseInt(e.target.value, 10)));

  const bgButton = document.getElementById("toggleBg");
  const bgColors = [
    new BABYLON.Color4(0.98, 0.98, 0.98, 1.0),
    new BABYLON.Color4(0.85, 0.85, 0.85, 1.0),
    new BABYLON.Color4(0.2, 0.2, 0.22, 1.0),
    new BABYLON.Color4(0.04, 0.05, 0.06, 1.0),
  ];
  // Match the constructor’s light gray default (index 1)
  let currentBgIndex = 1;
  bgButton.addEventListener("click", () => {
    currentBgIndex = (currentBgIndex + 1) % bgColors.length;
    viewer.scene.clearColor = bgColors[currentBgIndex];
  });

  const prevBtn = document.getElementById("prevCamBtn");
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
