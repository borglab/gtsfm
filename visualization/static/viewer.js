// visualization/static/viewer.js

class ColmapViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.04, 0.05, 0.06, 1.0);

    // ArcRotateCamera with sane defaults; we’ll sync pos/target then rebuild angles
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

    // Inputs & limits tuned to avoid “locked” feeling
    this.camera.angularSensibilityX = 1000;
    this.camera.angularSensibilityY = 1000;
    this.camera.panningSensibility = 1000;
    this.camera.wheelPrecision = 40;
    this.camera.lowerRadiusLimit = 0.1;
    this.camera.upperRadiusLimit = 10000;
    this.camera.lowerBetaLimit = 0.001;
    this.camera.upperBetaLimit = Math.PI - 0.001;

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
    this.ptSize = Math.max(1, Math.min(10, px));
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

    console.log(`Loaded: #points=${points.length}, #cams=${this.cameras.length}`);

    if (points.length) await this._createPCS(points);
    if (this.cameras.length) this._createFrusta(this.cameras);

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
        const centerWorld = BABYLON.Vector3.TransformNormal(t.scale(-1), Rcw);

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

    const mat = new BABYLON.PointsMaterial("pcmat", this.scene);
    mat.pointSize = this.ptSize;
    mat.disableLighting = true;
    mat.useVertexColor = true;
    this.pointsMesh.material = mat;
    this.pointsMesh.isPickable = false;

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
    const frustumScale = 0.02 * size;   // 2% of scene extent
    const pivotDiam = 0.006 * size;   // 0.6% sphere

    for (const c of cams) {
      const node = new BABYLON.TransformNode("camNode", this.scene);
      node.position.copyFrom(c.center);
      node.rotationQuaternion = BABYLON.Quaternion.FromRotationMatrix(c.R);
      node.parent = this.frustumGroup;

      const frustum = this._frustumLinesLocal(frustumScale);
      frustum.parent = node;

      const pivot = BABYLON.MeshBuilder.CreateSphere("camPivot", { diameter: pivotDiam }, this.scene);
      const pivotMat = new BABYLON.StandardMaterial("camPivotMat", this.scene);
      pivotMat.emissiveColor = new BABYLON.Color3(1, 0.35, 0.35);
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
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    if (this.pointsMesh) {
      const bb = this.pointsMesh.getBoundingInfo().boundingBox;
      minX = bb.minimumWorld.x; maxX = bb.maximumWorld.x;
      minY = bb.minimumWorld.y; maxY = bb.maximumWorld.y;
      minZ = bb.minimumWorld.z; maxZ = bb.maximumWorld.z;
    }

    if (minX === Infinity && this.cameras.length === 0) return;

    const sceneCenter = new BABYLON.Vector3(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2
    );

    this.camera.setTarget(sceneCenter);

    if (this.cameras.length > 0) {
      // Use first camera center as starting position
      const firstCamPos = this.cameras[0].center;
      this.camera.setPosition(firstCamPos);
      // Recompute alpha, beta, radius from pos/target to avoid “locked” rotation
      if (typeof this.camera.rebuildAnglesAndRadius === 'function') {
        this.camera.rebuildAnglesAndRadius();
      }
      // Do NOT immediately inflate radius; keep the view anchored to that position.
    } else {
      // No cameras => simple fit by radius
      const sceneSize = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
      this.camera.radius = sceneSize * 1.6;
    }
  }
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

  const renderList = (q = "") => {
    const needle = q.trim().toLowerCase();
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

  console.log('first camera center:', viewer.cameras[0]?.center);
  console.log('scene extent:', viewer._sceneExtent());

  renderList();
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
  let currentBgIndex = 3;
  bgButton.addEventListener("click", () => {
    currentBgIndex = (currentBgIndex + 1) % bgColors.length;
    viewer.scene.clearColor = bgColors[currentBgIndex];
  });
}

window.addEventListener("DOMContentLoaded", boot);