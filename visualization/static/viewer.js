class ColmapViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.04, 0.05, 0.06, 1.0);

    this.camera = new BABYLON.ArcRotateCamera("cam", -Math.PI/2, Math.PI/2.2, 10, new BABYLON.Vector3(0,0,0), this.scene);
    this.camera.attachControl(canvas, true);
    this.camera.lowerRadiusLimit = 0.05;
    this.camera.wheelPrecision = 40;

    this.light = new BABYLON.HemisphericLight("H", new BABYLON.Vector3(0,1,0), this.scene);
    this.light.intensity = 0.85;

    this.pcs = null;
    this.pointsMesh = null;
    this.frustumGroup = new BABYLON.TransformNode("frusta", this.scene);

    this.ptSize = 2;
    this.showCams = true;

    this.engine.runRenderLoop(() => this.scene.render());
    window.addEventListener("resize", () => this.engine.resize());
  }

  setPointSize(px) { this.ptSize = Math.max(1, Math.min(10, px)); if (this.pointsMesh && this.pointsMesh.material) this.pointsMesh.material.pointSize = this.ptSize; }
  setShowCams(v) { this.showCams = v; this.frustumGroup.setEnabled(!!v); }

  async loadScene({ pointsUrl, imagesUrl }) {
    await this._clear();

    const fetchOpts = { cache: "no-store" };
    const [pointsText, imagesText] = await Promise.all([
      fetch(pointsUrl, fetchOpts).then(r => {
        if (!r.ok) throw new Error("points3D.txt not found");
        return r.text();
      }),
      fetch(imagesUrl, fetchOpts).then(r => (r.ok ? r.text() : ""))
    ]);

    const points = this._parsePoints(pointsText);
    const cams = imagesText ? this._parseCams(imagesText) : [];
    if (cams.length) {
      console.log(`Loaded scene: ${points.length} points, ${cams.length} cameras from ${imagesUrl}`);
      console.debug("First camera center:", cams[0]?.center?.toString?.() ?? cams[0]?.center);
    } else {
      console.warn(`Loaded scene with ${points.length} points but no cameras found in ${imagesUrl}`);
    }

    if (points.length) await this._createPCS(points);
    if (cams.length) this._createFrusta(cams);

    this._autoFrame();
  }

  async _clear() {
    if (this.pointsMesh) { this.pointsMesh.dispose(); this.pointsMesh = null; }
    if (this.pcs) { this.pcs.dispose(); this.pcs = null; }
    this.frustumGroup.getChildren().forEach(n => n.dispose());
  }

  _parsePoints(text) {
    const pts = [];
    for (const line of text.split("\n")) {
      if (!line || line[0] === "#") continue;
      const s = line.trim().split(/\s+/);
      if (s.length >= 7) {
        pts.push({
          x: parseFloat(s[1]), y: -parseFloat(s[2]), z: parseFloat(s[3]), // flip Y to match Babylon’s Y-up
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
      const line = lines[i].trim();
      if (!line || line.startsWith("#")) continue;
      const s = line.split(/\s+/);
      if (s.length >= 10) {
        // COLMAP: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        const qw = parseFloat(s[1]), qx = parseFloat(s[2]), qy = parseFloat(s[3]), qz = parseFloat(s[4]);
        const tx = parseFloat(s[5]), ty = parseFloat(s[6]), tz = parseFloat(s[7]);

        // Rotation world->camera
        const qwc = new BABYLON.Quaternion(qx, qy, qz, qw);
        const Rwc = new BABYLON.Matrix();
        qwc.toRotationMatrix(Rwc);

        const Rcw = Rwc.transpose();
        const t = new BABYLON.Vector3(tx, ty, tz);

        // Camera center in world coordinates: C = -Rcw * t
        const centerWorld = BABYLON.Vector3.TransformNormal(t.scale(-1), Rcw);

        // Flip Y axis (COLMAP RH -> Babylon LH)
        const flip = BABYLON.Matrix.Scaling(1, -1, 1);
        const center = new BABYLON.Vector3(centerWorld.x, -centerWorld.y, centerWorld.z);
        const Rcam = flip.multiply(Rcw).multiply(flip);

        cams.push({ center, R: Rcam });

        // Skip 2D keypoints line if present
        if (i + 1 < lines.length && /^\d/.test(lines[i + 1])) i++;
      }
    }
    return cams;
  }


  async _createPCS(points) {
    const pcs = new BABYLON.PointsCloudSystem("pcs", { capacity: points.length, useColor: true }, this.scene);
    let idx = 0;
    pcs.addPoints(points.length, particle => {
      const s = points[idx++];
      particle.position.set(s.x, s.y, s.z);
      particle.color = new BABYLON.Color4(s.r, s.g, s.b, 1.0);
    });
    this.pointsMesh = await pcs.buildMeshAsync();
    const mat = new BABYLON.PointsMaterial("pcmat", this.scene);
    mat.pointSize = this.ptSize;
    mat.disableLighting = true;
    mat.useVertexColor = true;
    this.pointsMesh.material = mat;
    this.pointsMesh.alwaysSelectAsActiveMesh = true;
    this.pointsMesh.isPickable = false;
    this.pcs = pcs;
  }

  _createFrusta(cams) {
    for (const c of cams) {
      const node = new BABYLON.TransformNode("camNode", this.scene);
      node.position.copyFrom(c.center);
      node.rotationQuaternion = BABYLON.Quaternion.FromRotationMatrix(c.R);
      node.parent = this.frustumGroup;

      const frustum = this._frustumLinesLocal(0.6);
      frustum.parent = node;

      const pivot = BABYLON.MeshBuilder.CreateSphere("camPivot", { diameter: 0.06 }, this.scene);
      const pivotMat = new BABYLON.StandardMaterial("camPivotMat", this.scene);
      pivotMat.emissiveColor = new BABYLON.Color3(1, 0.35, 0.35);
      pivotMat.disableLighting = true;
      pivot.material = pivotMat;
      pivot.isPickable = false;
      pivot.parent = node;

      const axes = this._cameraAxesLocal(0.45);
      axes.forEach(axis => axis.parent = node);
    }

    this.setShowCams(this.showCams);
  }

  _frustumLinesLocal(scale) {
    const origin = BABYLON.Vector3.Zero();
    const corners = [
      new BABYLON.Vector3(-0.4, -0.3, 1),
      new BABYLON.Vector3( 0.4, -0.3, 1),
      new BABYLON.Vector3( 0.4,  0.3, 1),
      new BABYLON.Vector3(-0.4,  0.3, 1)
    ].map(v => v.scale(scale));

    const lines = [
      [origin, corners[0]],
      [origin, corners[1]],
      [origin, corners[2]],
      [origin, corners[3]],
      [corners[0], corners[1]],
      [corners[1], corners[2]],
      [corners[2], corners[3]],
      [corners[3], corners[0]]
    ];
    const frustum = BABYLON.MeshBuilder.CreateLineSystem(
      "frustumLines",
      { lines, useVertexAlpha: false, updatable: false },
      this.scene
    );
    frustum.color = new BABYLON.Color3(1, 0.45, 0.25);
    frustum.disableLighting = true;
    frustum.isPickable = false;
    frustum.alwaysSelectAsActiveMesh = true;
    return frustum;
  }

  _cameraAxesLocal(scale) {
    const axes = [];
    const colors = [
      new BABYLON.Color3(1, 0.15, 0.15),
      new BABYLON.Color3(0.2, 1.0, 0.2),
      new BABYLON.Color3(0.25, 0.45, 1.0)
    ];
    const directions = [
      new BABYLON.Vector3(scale, 0, 0),
      new BABYLON.Vector3(0, scale, 0),
      new BABYLON.Vector3(0, 0, scale)
    ];

    directions.forEach((dir, idx) => {
      const line = BABYLON.MeshBuilder.CreateLines(
        `camAxis${idx}`,
        { points: [BABYLON.Vector3.Zero(), dir] },
        this.scene
      );
      line.color = colors[idx];
      line.disableLighting = true;
      line.alwaysSelectAsActiveMesh = true;
      line.isPickable = false;
      axes.push(line);
    });
    return axes;
  }

  _autoFrame() {
    let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
    if (this.pointsMesh) {
      const bb = this.pointsMesh.getBoundingInfo().boundingBox;
      const mn = bb.minimum, mx = bb.maximum;
      minX = Math.min(minX, mn.x); maxX = Math.max(maxX, mx.x);
      minY = Math.min(minY, mn.y); maxY = Math.max(maxY, mx.y);
      minZ = Math.min(minZ, mn.z); maxZ = Math.max(maxZ, mx.z);
    }
    this.frustumGroup.getChildren().forEach(f => {
      const p = f.position;
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    });
    if (minX === Infinity) return;
    const center = new BABYLON.Vector3((minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2);
    const size = Math.max(maxX-minX, maxY-minY, maxZ-minZ) || 1.0;
    this.camera.setTarget(center);
    this.camera.radius = size * 1.6;
  }
}

async function boot() {
  const info = document.getElementById("info");
  const listEl = document.getElementById("sceneList");
  const filterEl = document.getElementById("filter");
  const canvas = document.getElementById("renderCanvas");
  const viewer = new ColmapViewer(canvas);

  const data = await fetch("/api/scenes").then(r=>r.json());
  info.textContent = `${data.count} reconstructions found under ${data.base_dir}`;

  let items = data.items;
  const renderList = (q="") => {
    listEl.innerHTML = "";
    const needle = q.trim().toLowerCase();
    items.filter(it => it.label.toLowerCase().includes(needle)).forEach((it, idx) => {
      const el = document.createElement("div");
      el.className = "item";
      el.dataset.idx = String(idx);
      el.innerHTML = `<div>${it.label}</div><small>${it.rel_path}</small>`;
      el.onclick = async () => {
        listEl.querySelectorAll(".item").forEach(n => n.classList.remove("active"));
        el.classList.add("active");
        await viewer.loadScene({ pointsUrl: it.points, imagesUrl: it.images });
      };
      listEl.appendChild(el);
    });
  };
  renderList();

  // auto-load first if present
  if (items.length > 0) {
    const first = listEl.querySelector(".item");
    if (first) first.click();
  }

  filterEl.addEventListener("input", (e) => renderList(filterEl.value));

  // HUD controls
  document.getElementById("toggleCams").addEventListener("change", (e) => {
    viewer.setShowCams(e.target.checked);
  });
  document.getElementById("ptSize").addEventListener("input", (e) => {
    viewer.setPointSize(parseInt(e.target.value, 10));
  });
}

window.addEventListener("DOMContentLoaded", boot);
