import * as THREE from 'three';

export class CameraEngine {
    constructor(camera, orbitControls) {
        this.camera = camera;
        this.orbitControls = orbitControls;
        this.flythrough = null;
    }

    flyTo(position, target, duration = 1.5) {
        this.flythrough = {
            startPos: this.camera.position.clone(),
            endPos: position.clone(),
            startTarget: this.orbitControls.target.clone(),
            endTarget: target.clone(),
            startTime: performance.now(),
            duration: duration * 1000
        };
    }

    stopFlythrough() {
        this.flythrough = null;
    }

    update(time) {
        if (!this.flythrough) return;

        const f = this.flythrough;
        const t = Math.min((performance.now() - f.startTime) / f.duration, 1);
        const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        this.camera.position.lerpVectors(f.startPos, f.endPos, e);
        this.orbitControls.target.lerpVectors(f.startTarget, f.endTarget, e);
        this.orbitControls.update();

        if (t >= 1) {
            this.flythrough = null;
        }
    }
}
