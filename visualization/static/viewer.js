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

    this.ptSize = 2;
    this.showCams = true;

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

  async loadScene({ pointsUrl, imagesUrl }) {
    await this._clear();

    const fetchOpts = { cache: "no-store" };
    const [pointsText, imagesText] = await Promise.all([
      fetch(pointsUrl, fetchOpts).then(r => r.ok ? r.text() : Promise.reject(new Error(`Failed to load ${pointsUrl}`))),
      fetch(imagesUrl, fetchOpts).then(r => r.ok ? r.text() : "")
    ]);

    const points = this._parsePoints(pointsText);
    this.cameras = imagesText ? this._parseCams(imagesText) : [];
    console.log('first camera center:', this.cameras[0]?.center);
    console.log('scene extent:', this._sceneExtent());
    console.log(`Loaded: #points=${points.length}, #cams=${this.cameras.length}`);

    if (points.length) await this._createPCS(points);
    if (this.cameras.length) this._createFrusta(this.cameras);

    // centroid from a 100-point sample (fallback to bbox center if needed)
    this.sceneCenter = points.length
      ? this._centroidSample(points, 100)
      : (this.pointsMesh
        ? this.pointsMesh.getBoundingInfo().boundingBox.centerWorld.clone()
        : new BABYLON.Vector3(0, 0, 0));

    // remember which camera we're on for next/prev UI
    this.camIndex = 0;

    this._autoFrame();
  }

  async _clear() {
    this.cameras = [];

    if (this.pointsMesh) { this.pointsMesh.dispose(); this.pointsMesh = null; }
    if (this.pcs) { this.pcs.dispose(); this.pcs = null; }

    if (this.frustumGroup) {
      for (const ch of this.frustumGroup.getChildren()) ch.dispose();
      this.frustumGroup.setEnabled(false);
      this.frustumGroup.setEnabled(true);
    }
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

        cams.push({ center, R: Rcam });

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
  }

  nextCam() { this._gotoCam((this.camIndex ?? 0) + 1); }
  prevCam() { this._gotoCam((this.camIndex ?? 0) - 1); }
}

// ------------------- app bootstrap -------------------

async function boot() {
  const info = document.getElementById("info");
  const listEl = document.getElementById("sceneList");
  const filterEl = document.getElementById("filter");
  const canvas = document.getElementById("renderCanvas");
  const viewer = new ColmapViewer(canvas);

  let allItems = [];

  try {
    const data = await fetch("/api/scenes").then(r => r.ok ? r.json() : Promise.reject(`Fetch failed: ${r.statusText}`));
    info.textContent = `${data.count} reconstructions found in ${data.base_dir}`;
    allItems = data.items.map((item, index) => ({ ...item, originalIndex: index }));
  } catch (error) {
    info.textContent = "Error loading reconstruction data.";
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
            <div class="item" data-idx="${item.originalIndex}">
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
      listEl.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b;">No scenes found.</div>';
      return;
    }
    const sceneTree = buildTree(filtered);
    listEl.innerHTML = renderTree(sceneTree);
    listEl.querySelectorAll(".item").forEach(el => {
      el.onclick = async () => {
        listEl.querySelectorAll(".item").forEach(n => n.classList.remove("active"));
        el.classList.add("active");
        const originalIndex = parseInt(el.dataset.idx, 10);
        const item = allItems.find(it => it.originalIndex === originalIndex);
        if (item) {
          console.log("Loading scene:", item);
          await viewer.loadScene({ pointsUrl: item.points, imagesUrl: item.images });
        }
      };
    });
  };


  // Default search term
  filterEl.value = "ba_output";
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
}

window.addEventListener("DOMContentLoaded", boot);