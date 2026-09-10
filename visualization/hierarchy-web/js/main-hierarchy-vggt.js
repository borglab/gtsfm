import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VGGTDataLoader } from './data-loader-vggt.js?v=17';
import { SquarenessLayoutEngine } from './layout-engine-squareness.js?v=17';
import { InteractionEngine } from './interaction-engine.js?v=4';
import { SquarenessAnimationEngine } from './animation-engine-squareness.js?v=16';
import { CameraEngine } from './camera-engine.js?v=4';

class VGGTHierarchyApp {
    constructor() {
        this.initThree();
        this.initUI();
    }

    initThree() {
        this.container = document.getElementById('canvas-container');
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        
        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 50000);
        this.camera.position.set(0, 0, 200);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.container.appendChild(this.renderer.domElement);
        
        this.orbitControls = new OrbitControls(this.camera, this.renderer.domElement);
        this.orbitControls.enableDamping = true;
        this.orbitControls.dampingFactor = 0.05;
        this.orbitControls.autoRotate = false;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        this.scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(10, 10, 10);
        this.scene.add(dirLight);

        this.worldGroup = new THREE.Group();
        this.scene.add(this.worldGroup);

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    initUI() {
        this.ui = {
            loading: document.getElementById('loading'),
            loadingText: document.querySelector('.loading-text'),
            eventLabel: document.getElementById('event-label'),
            progressBar: document.getElementById('timeline-progress'),
            stats: document.getElementById('stats-display'),
            prevBtn: document.getElementById('btn-prev'),
            nextBtn: document.getElementById('btn-next'),
            playBtn: document.getElementById('btn-play'),
            resetBtn: document.getElementById('btn-reset'),
            track: document.getElementById('timeline-track')
        };

        this.ui.prevBtn.addEventListener('click', () => this.step(-1));
        this.ui.nextBtn.addEventListener('click', () => this.step(1));
        this.ui.resetBtn.addEventListener('click', () => this.reset());
        this.ui.playBtn.addEventListener('click', () => this.togglePlay());
        
        this.ui.track.addEventListener('click', (e) => {
            if (!this.animationEngine) return;
            const rect = this.ui.track.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            const index = Math.floor(pct * this.animationEngine.mergeEvents.length);
            this.jumpTo(index);
        });

        this.ui.recordBtn = document.getElementById('btn-record');
        this.ui.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.mediaRecorder = null;
        this.recordedChunks = [];
    }

    async start() {
        try {
            this.dataLoader = new VGGTDataLoader();
            this.dataLoader.onProgress = (loaded, total) => {
                this.ui.loadingText.textContent = `Loading VGGT Clusters... ${loaded}/${total}`;
            };

            const clusters = await this.dataLoader.load();
            
            let loadedCount = 0;
            for (const c of clusters.values()) {
                if (c.pointCloud) loadedCount++;
            }
            
            if (loadedCount === 0) {
                throw new Error("No point clouds loaded. Check that data/ directory exists.");
            }

            console.log(`Loaded ${loadedCount}/${clusters.size} clusters with point data`);
        
            for (const cluster of clusters.values()) {
                this.worldGroup.add(cluster.group);
            }

            this.layoutEngine = new SquarenessLayoutEngine(clusters);
            this.layoutEngine.computeLayout();

            this.fitCameraToLayout();

            this.animationEngine = new SquarenessAnimationEngine(clusters, this.layoutEngine, this.worldGroup);
            this.events = this.animationEngine.initTimeline();
            this.currentEventIndex = 0;

            this.interactionEngine = new InteractionEngine(
                this.camera, 
                this.renderer.domElement, 
                clusters, 
                this.orbitControls
            );

            this.cameraEngine = new CameraEngine(this.camera, this.orbitControls);

            if (this.events.length > 0) {
                this.animationEngine.applyEventInstant(0);
            }
            this.updateUI();

            this.ui.loading.style.display = 'none';
            
            this.animate();
            
        } catch (err) {
            console.error("App Start Error:", err);
            this.ui.loadingText.innerHTML = `<span style="color: #ff4444">Error starting app:<br>${err.message}</span>`;
        }
    }

    fitCameraToLayout() {
        const b = this.layoutEngine.bounds;
        if (!b || b.width === undefined || b.height === undefined) {
            console.warn("No layout bounds, using defaults");
            this.camera.position.set(0, 0, 200);
            this.orbitControls.target.set(0, 0, 0);
            this.orbitControls.update();
            return;
        }
        
        const fov = this.camera.fov;
        const aspect = window.innerWidth / window.innerHeight;
        
        const vFovRad = THREE.MathUtils.degToRad(fov / 2);
        const hFovRad = Math.atan(aspect * Math.tan(vFovRad));
        
        const distForHeight = (b.height / 2) / Math.tan(vFovRad);
        const distForWidth = (b.width / 2) / Math.tan(hFovRad);
        
        let dist = Math.max(distForHeight, distForWidth) * 1.1;
        dist = Math.max(dist, 8);
        
        const centerY = (b.maxY + b.minY) / 2;
        const centerX = (b.maxX + b.minX) / 2;
        
        this.camera.position.set(centerX, centerY, dist);
        this.camera.lookAt(centerX, centerY, 0);
        this.orbitControls.target.set(centerX, centerY, 0);
        this.orbitControls.update();
    }

    step(direction) {
        if (direction > 0) {
            if (this.currentEventIndex < this.events.length - 1) {
                this.currentEventIndex++;
                this.animationEngine.playEvent(this.currentEventIndex, 1);
            }
        } else {
            if (this.currentEventIndex > 0) {
                this.animationEngine.playEvent(this.currentEventIndex, -1);
                this.currentEventIndex--;
            }
        }
        this.updateUI();
    }
    
    jumpTo(index) {
        if (index < 0) index = 0;
        if (index >= this.events.length) index = this.events.length - 1;
        this.currentEventIndex = index;
        this.animationEngine.applyEventInstant(index);
        this.updateUI();
    }

    reset() {
        this.isPlaying = false;
        this.ui.playBtn.textContent = 'Play';
        this.cameraEngine.stopFlythrough();
        this.jumpTo(0);
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.ui.playBtn.textContent = this.isPlaying ? 'Pause' : 'Play';
        
        if (this.isPlaying && this.currentEventIndex >= this.events.length - 1) {
            this.jumpTo(0);
        }
    }

    toggleRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.ui.recordBtn.textContent = 'Record';
            this.ui.recordBtn.classList.remove('recording');
            return;
        }

        this.recordedChunks = [];
        const canvas = this.renderer.domElement;
        const stream = canvas.captureStream(30);
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9',
            videoBitsPerSecond: 5000000
        });

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.recordedChunks.push(e.data);
        };

        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            a.download = `gerrard-hall-recording-${timestamp}.webm`;
            a.click();
            URL.revokeObjectURL(url);
        };

        this.mediaRecorder.start();
        this.ui.recordBtn.textContent = 'Stop';
        this.ui.recordBtn.classList.add('recording');
    }

    updateUI() {
        const count = this.events.length;
        if (count === 0) return;
        
        const progress = (this.currentEventIndex / (count - 1)) * 100;
        this.ui.progressBar.style.width = `${progress}%`;
        
        const event = this.events[this.currentEventIndex];
        const eventType = event.isLeaf ? 'VGGT' : 'Merge';
        this.ui.eventLabel.textContent = `Event ${this.currentEventIndex + 1}/${count}: ${eventType} — ${event.path}`;
        
        let visiblePoints = 0;
        let visibleClusters = 0;
        for (const c of this.dataLoader.clusters.values()) {
            if (c.pointCloud && c.pointCloud.visible) {
                visibleClusters++;
                visiblePoints += c.pointsCount;
            }
        }
        if (this.animationEngine && this.animationEngine.transitionClouds.length > 0) {
            for (const tc of this.animationEngine.transitionClouds) {
                if (tc.visible && tc.geometry && tc.geometry.attributes.position) {
                    visiblePoints += tc.geometry.attributes.position.count;
                    visibleClusters++;
                }
            }
        }
        this.ui.stats.textContent = `Clusters: ${visibleClusters} | Points: ${visiblePoints.toLocaleString()}`;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        const time = performance.now() / 1000;
        const dt = 0.016;

        if (this.isPlaying) {
            if (!this.lastStepTime) this.lastStepTime = time;
            const hasActiveAnims = this.animationEngine && this.animationEngine.activeAnimations.length > 0;
            if (!hasActiveAnims && time - this.lastStepTime > 1.0) {
                if (this.currentEventIndex < this.events.length - 1) {
                    this.step(1);
                    this.lastStepTime = time;
                } else {
                    this.togglePlay();
                }
            }
        } else {
            this.lastStepTime = 0;
        }

        if (this.animationEngine) {
            const hadAnimations = this.animationEngine.activeAnimations.length > 0;
            this.animationEngine.update(dt);
            if (hadAnimations) this.updateUI();
        }
        if (this.cameraEngine) this.cameraEngine.update(time);
        
        this.orbitControls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

const app = new VGGTHierarchyApp();
window.app = app;
app.start();
