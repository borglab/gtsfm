import * as THREE from 'three';

/**
 * Animation Engine with Per-Point Interpolation
 *
 * Merge transitions move individual points from their child-cluster
 * positions to their merged-result positions, matching the approach
 * from the Gemini point-set interpolation demo:
 *   - Matched points lerp smoothly (cubic easing)
 *   - Child-only points fade out in place
 *   - Merged-only points fade in at target
 */
export class SquarenessAnimationEngine {
    constructor(clusters, layoutEngine, worldGroup) {
        this.clusters = clusters;
        this.layoutEngine = layoutEngine;
        this.worldGroup = worldGroup;
        this.mergeEvents = [];
        this.activeAnimations = [];
        this.transitionClouds = [];
        this.mergeDuration = 2.5;
        this.leafFadeDuration = 0.5;
    }

    initTimeline() {
        const treeNodes = this.layoutEngine.treeNodes;
        if (!treeNodes || treeNodes.length === 0) {
            console.warn("No tree nodes for timeline");
            return [];
        }

        const leaves = [];
        const merges = [];
        for (const node of treeNodes) {
            if (node.children.length === 0) leaves.push(node);
            else merges.push(node);
        }

        leaves.sort((a, b) => {
            if (b.depth !== a.depth) return b.depth - a.depth;
            return a.cluster.path.localeCompare(b.cluster.path);
        });
        merges.sort((a, b) => {
            if (b.depth !== a.depth) return b.depth - a.depth;
            return a.cluster.path.localeCompare(b.cluster.path);
        });

        for (const node of leaves) {
            this.mergeEvents.push({
                path: node.cluster.path,
                cluster: node.cluster,
                isLeaf: true,
                children: [],
                depth: node.depth
            });
        }
        for (const node of merges) {
            this.mergeEvents.push({
                path: node.cluster.path,
                cluster: node.cluster,
                isLeaf: false,
                children: node.children.map(c => c.cluster.path),
                depth: node.depth
            });
        }

        console.log(`Timeline: ${this.mergeEvents.length} events (${leaves.length} leaves + ${merges.length} merges)`);
        return this.mergeEvents;
    }

    cleanupTransitions() {
        for (const cloud of this.transitionClouds) {
            this.worldGroup.remove(cloud);
            cloud.geometry.dispose();
            cloud.material.dispose();
        }
        this.transitionClouds = [];
        this.activeAnimations = this.activeAnimations.filter(a => a.type !== 'mergeTransition');
    }

    applyEventInstant(eventIndex) {
        this.cleanupTransitions();
        this.activeAnimations = [];

        for (const cluster of this.clusters.values()) {
            if (cluster.pointCloud) {
                cluster.pointCloud.visible = false;
                cluster.pointCloud.material.opacity = 1;
            }
            if (cluster.hierarchyPosition) {
                cluster.group.position.copy(cluster.hierarchyPosition);
            }
            if (cluster.fitScale) {
                cluster.group.scale.setScalar(cluster.fitScale);
            }
        }

        for (let i = 0; i <= eventIndex; i++) {
            const evt = this.mergeEvents[i];
            if (!evt) continue;
            const c = evt.cluster;

            if (evt.isLeaf) {
                if (c.pointCloud) c.pointCloud.visible = true;
            } else {
                if (c.pointCloud) c.pointCloud.visible = true;
                for (const childPath of evt.children) {
                    const child = this.clusters.get(childPath);
                    if (child && child.pointCloud) child.pointCloud.visible = false;
                }
            }
        }
    }

    playEvent(eventIndex, direction = 1) {
        const evt = this.mergeEvents[eventIndex];
        if (!evt) return;

        if (direction > 0) {
            if (evt.isLeaf) {
                if (evt.cluster.pointCloud) {
                    evt.cluster.pointCloud.visible = true;
                    this.animateFadeIn(evt.cluster);
                }
            } else {
                this.playMergeTransition(evt);
            }
        } else {
            this.cleanupTransitions();
            if (evt.isLeaf) {
                if (evt.cluster.pointCloud) this.animateFadeOut(evt.cluster);
            } else {
                if (evt.cluster.pointCloud) {
                    evt.cluster.pointCloud.visible = false;
                    evt.cluster.pointCloud.material.opacity = 1;
                }
                for (const childPath of evt.children) {
                    const child = this.clusters.get(childPath);
                    if (child && child.pointCloud) {
                        child.pointCloud.visible = true;
                        child.pointCloud.material.opacity = 1;
                        child.group.position.copy(child.hierarchyPosition);
                        if (child.fitScale) child.group.scale.setScalar(child.fitScale);
                    }
                }
            }
        }
    }

    playMergeTransition(evt) {
        const merged = evt.cluster;
        const matchData = merged.matchData;

        if (!matchData || !merged.pointCloud) {
            if (merged.pointCloud) {
                merged.pointCloud.visible = true;
                this.animateFadeIn(merged);
            }
            for (const childPath of evt.children) {
                const child = this.clusters.get(childPath);
                if (child && child.pointCloud && child.pointCloud.visible) {
                    this.animateFallbackMerge(child, merged.hierarchyPosition);
                }
            }
            return;
        }

        const { matchedPairs, childOnlyPoints, mergedOnlyIndices } = matchData;

        for (const childPath of evt.children) {
            const child = this.clusters.get(childPath);
            if (child && child.pointCloud) child.pointCloud.visible = false;
        }
        merged.pointCloud.visible = false;

        // --- Matched points cloud ---
        const mLen = matchedPairs.length;
        const matchedStart = new Float32Array(mLen * 3);
        const matchedEnd = new Float32Array(mLen * 3);
        const matchedColors = new Float32Array(mLen * 3);

        for (let i = 0; i < mLen; i++) {
            const pair = matchedPairs[i];
            const child = this.clusters.get(pair.childPath);
            if (!child || !child.pointCloud) continue;

            this.writeWorldPos(child, pair.childIdx, matchedStart, i * 3);
            this.writeWorldPos(merged, pair.mergedIdx, matchedEnd, i * 3);
            this.writeColor(child, pair.childIdx, matchedColors, i * 3);
        }

        const matchedGeom = new THREE.BufferGeometry();
        matchedGeom.setAttribute('position', new THREE.Float32BufferAttribute(matchedStart.slice(), 3));
        matchedGeom.setAttribute('color', new THREE.Float32BufferAttribute(matchedColors, 3));
        const matchedMat = this.makeTransitionMaterial(1.0);
        const matchedCloud = new THREE.Points(matchedGeom, matchedMat);

        // --- Child-only cloud (fly to nearest merged point) ---
        const coLen = childOnlyPoints.length;
        const coStart = new Float32Array(coLen * 3);
        const coEnd = new Float32Array(coLen * 3);
        const coCol = new Float32Array(coLen * 3);

        for (let i = 0; i < coLen; i++) {
            const cp = childOnlyPoints[i];
            const child = this.clusters.get(cp.childPath);
            if (!child || !child.pointCloud) continue;
            this.writeWorldPos(child, cp.childIdx, coStart, i * 3);
            this.writeColor(child, cp.childIdx, coCol, i * 3);
            if (cp.flyToMergedIdx !== undefined) {
                this.writeWorldPos(merged, cp.flyToMergedIdx, coEnd, i * 3);
            } else {
                coEnd[i * 3] = coStart[i * 3];
                coEnd[i * 3 + 1] = coStart[i * 3 + 1];
                coEnd[i * 3 + 2] = coStart[i * 3 + 2];
            }
        }

        const coGeom = new THREE.BufferGeometry();
        coGeom.setAttribute('position', new THREE.Float32BufferAttribute(coStart.slice(), 3));
        coGeom.setAttribute('color', new THREE.Float32BufferAttribute(coCol, 3));
        const coMat = this.makeTransitionMaterial(1.0);
        const childOnlyCloud = new THREE.Points(coGeom, coMat);

        // --- Merged-only cloud (fly from nearest child point) ---
        const moLen = mergedOnlyIndices.length;
        const moStart = new Float32Array(moLen * 3);
        const moEnd = new Float32Array(moLen * 3);
        const moCol = new Float32Array(moLen * 3);

        for (let i = 0; i < moLen; i++) {
            const entry = mergedOnlyIndices[i];
            const mIdx = typeof entry === 'object' ? entry.mergedIdx : entry;
            this.writeWorldPos(merged, mIdx, moEnd, i * 3);
            this.writeColor(merged, mIdx, moCol, i * 3);
            if (typeof entry === 'object' && entry.flyFromChildPath) {
                const srcChild = this.clusters.get(entry.flyFromChildPath);
                if (srcChild && srcChild.pointCloud) {
                    this.writeWorldPos(srcChild, entry.flyFromChildIdx, moStart, i * 3);
                } else {
                    moStart[i * 3] = moEnd[i * 3];
                    moStart[i * 3 + 1] = moEnd[i * 3 + 1];
                    moStart[i * 3 + 2] = moEnd[i * 3 + 2];
                }
            } else {
                moStart[i * 3] = moEnd[i * 3];
                moStart[i * 3 + 1] = moEnd[i * 3 + 1];
                moStart[i * 3 + 2] = moEnd[i * 3 + 2];
            }
        }

        const moGeom = new THREE.BufferGeometry();
        moGeom.setAttribute('position', new THREE.Float32BufferAttribute(moStart.slice(), 3));
        moGeom.setAttribute('color', new THREE.Float32BufferAttribute(moCol, 3));
        const moMat = this.makeTransitionMaterial(0.0);
        const mergedOnlyCloud = new THREE.Points(moGeom, moMat);

        this.worldGroup.add(matchedCloud);
        this.worldGroup.add(childOnlyCloud);
        this.worldGroup.add(mergedOnlyCloud);
        this.transitionClouds.push(matchedCloud, childOnlyCloud, mergedOnlyCloud);

        this.activeAnimations.push({
            type: 'mergeTransition',
            matchedCloud, childOnlyCloud, mergedOnlyCloud,
            matchedStart, matchedEnd,
            matchedCount: mLen,
            coStart, coEnd, coCount: coLen,
            moStart, moEnd, moCount: moLen,
            mergedCluster: merged,
            childPaths: evt.children,
            startTime: performance.now(),
            duration: this.mergeDuration * 1000,
            onComplete: () => {
                this.worldGroup.remove(matchedCloud);
                this.worldGroup.remove(childOnlyCloud);
                this.worldGroup.remove(mergedOnlyCloud);
                matchedGeom.dispose(); matchedMat.dispose();
                coGeom.dispose(); coMat.dispose();
                moGeom.dispose(); moMat.dispose();
                this.transitionClouds = this.transitionClouds.filter(
                    c => c !== matchedCloud && c !== childOnlyCloud && c !== mergedOnlyCloud
                );
                merged.pointCloud.visible = true;
                merged.pointCloud.material.opacity = 1;
            }
        });
    }

    writeWorldPos(cluster, localIdx, out, offset) {
        const pos = cluster.pointCloud.geometry.attributes.position;
        const s = cluster.fitScale || 1;
        const hp = cluster.hierarchyPosition;
        out[offset]     = hp.x + pos.getX(localIdx) * s;
        out[offset + 1] = hp.y + pos.getY(localIdx) * s;
        out[offset + 2] = hp.z + pos.getZ(localIdx) * s;
    }

    writeColor(cluster, localIdx, out, offset) {
        const col = cluster.pointCloud.geometry.attributes.color;
        out[offset]     = col.getX(localIdx);
        out[offset + 1] = col.getY(localIdx);
        out[offset + 2] = col.getZ(localIdx);
    }

    makeTransitionMaterial(opacity) {
        return new THREE.PointsMaterial({
            size: 5.0,
            vertexColors: true,
            sizeAttenuation: false,
            transparent: true,
            opacity,
            depthWrite: false
        });
    }

    animateFadeIn(cluster) {
        if (!cluster.pointCloud) return;
        cluster.pointCloud.material.opacity = 0;
        cluster.pointCloud.material.transparent = true;
        this.activeAnimations.push({
            type: 'fadeIn', cluster,
            startTime: performance.now(),
            duration: this.leafFadeDuration * 1000
        });
    }

    animateFadeOut(cluster) {
        if (!cluster.pointCloud) return;
        cluster.pointCloud.material.transparent = true;
        this.activeAnimations.push({
            type: 'fadeOut', cluster,
            startTime: performance.now(),
            duration: this.leafFadeDuration * 1000,
            onComplete: () => {
                cluster.pointCloud.visible = false;
                cluster.pointCloud.material.opacity = 1;
            }
        });
    }

    animateFallbackMerge(child, targetPos) {
        if (!child.pointCloud || !targetPos) return;
        const startPos = child.group.position.clone();
        const endPos = targetPos.clone();
        const startScale = child.group.scale.x;
        this.activeAnimations.push({
            type: 'fallbackMerge', cluster: child,
            startPos, endPos, startScale,
            startTime: performance.now(),
            duration: this.mergeDuration * 1000,
            onComplete: () => {
                child.pointCloud.visible = false;
                child.pointCloud.material.opacity = 1;
                child.group.position.copy(child.hierarchyPosition);
                child.group.scale.setScalar(startScale);
            }
        });
    }

    update(dt) {
        const now = performance.now();
        for (let i = this.activeAnimations.length - 1; i >= 0; i--) {
            const a = this.activeAnimations[i];
            const t = Math.min((now - a.startTime) / a.duration, 1);
            const e = this.easeInOutCubic(t);

            switch (a.type) {
                case 'fadeIn':
                    a.cluster.pointCloud.material.opacity = e;
                    break;

                case 'fadeOut':
                    a.cluster.pointCloud.material.opacity = 1 - e;
                    break;

                case 'fallbackMerge':
                    a.cluster.group.position.lerpVectors(a.startPos, a.endPos, e);
                    a.cluster.group.scale.setScalar(a.startScale * (1 - e * 0.5));
                    a.cluster.pointCloud.material.opacity = 1 - e * 0.85;
                    break;

                case 'mergeTransition': {
                    const mArr = a.matchedCloud.geometry.attributes.position.array;
                    for (let j = 0; j < a.matchedCount; j++) {
                        const j3 = j * 3;
                        mArr[j3]     = a.matchedStart[j3]     + (a.matchedEnd[j3]     - a.matchedStart[j3])     * e;
                        mArr[j3 + 1] = a.matchedStart[j3 + 1] + (a.matchedEnd[j3 + 1] - a.matchedStart[j3 + 1]) * e;
                        mArr[j3 + 2] = a.matchedStart[j3 + 2] + (a.matchedEnd[j3 + 2] - a.matchedStart[j3 + 2]) * e;
                    }
                    a.matchedCloud.geometry.attributes.position.needsUpdate = true;

                    const coArr = a.childOnlyCloud.geometry.attributes.position.array;
                    for (let j = 0; j < a.coCount; j++) {
                        const j3 = j * 3;
                        coArr[j3]     = a.coStart[j3]     + (a.coEnd[j3]     - a.coStart[j3])     * e;
                        coArr[j3 + 1] = a.coStart[j3 + 1] + (a.coEnd[j3 + 1] - a.coStart[j3 + 1]) * e;
                        coArr[j3 + 2] = a.coStart[j3 + 2] + (a.coEnd[j3 + 2] - a.coStart[j3 + 2]) * e;
                    }
                    a.childOnlyCloud.geometry.attributes.position.needsUpdate = true;
                    a.childOnlyCloud.material.opacity = 1 - e * 0.3;

                    const moArr = a.mergedOnlyCloud.geometry.attributes.position.array;
                    for (let j = 0; j < a.moCount; j++) {
                        const j3 = j * 3;
                        moArr[j3]     = a.moStart[j3]     + (a.moEnd[j3]     - a.moStart[j3])     * e;
                        moArr[j3 + 1] = a.moStart[j3 + 1] + (a.moEnd[j3 + 1] - a.moStart[j3 + 1]) * e;
                        moArr[j3 + 2] = a.moStart[j3 + 2] + (a.moEnd[j3 + 2] - a.moStart[j3 + 2]) * e;
                    }
                    a.mergedOnlyCloud.geometry.attributes.position.needsUpdate = true;
                    a.mergedOnlyCloud.material.opacity = e;
                    break;
                }
            }

            if (t >= 1) {
                if (a.onComplete) a.onComplete();
                this.activeAnimations.splice(i, 1);
            }
        }
    }

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
}
