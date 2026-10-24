import * as THREE from 'three';

/**
 * Squareness-Based Recursive Rectangle Layout Engine
 * 
 * Only LEAF nodes get spatial positions. Merged (non-leaf) nodes
 * get a mergeTargetPosition and mergeRegion used during merge animations.
 */
export class SquarenessLayoutEngine {
    constructor(clusters) {
        this.clusters = clusters;
        this.rootCluster = clusters.get('merged');
        this.bounds = null;
        this.treeNodes = [];
        this.PADDING_FRAC = 0.015;
    }

    compositions(n) {
        const result = [];
        const generate = (remaining, current) => {
            if (remaining === 0) { result.push([...current]); return; }
            for (let first = 1; first <= remaining; first++) {
                current.push(first);
                generate(remaining - first, current);
                current.pop();
            }
        };
        generate(n, []);
        return result;
    }

    squarenessForRow(W, H, n, m) {
        const rho = (H * m * m) / (W * n);
        return rho <= 1 ? rho : 1 / rho;
    }

    bestEqualAreaPartition(W, H, n) {
        if (n === 1) {
            return {
                squareness: Math.min(W, H) / Math.max(W, H),
                mode: 'rows', groups: [1],
                layout: [{ x: 0, y: 0, w: W, h: H }]
            };
        }

        const allComps = this.compositions(n);
        let bestS = -1, bestMode = 'rows', bestGroups = null;

        for (const ms of allComps) {
            const s = Math.min(...ms.map(m => this.squarenessForRow(W, H, n, m)));
            if (s > bestS) { bestS = s; bestMode = 'rows'; bestGroups = ms; }
        }
        for (const ms of allComps) {
            const s = Math.min(...ms.map(m => this.squarenessForRow(H, W, n, m)));
            if (s > bestS) { bestS = s; bestMode = 'cols'; bestGroups = ms; }
        }

        const layout = this.buildTiles(W, H, bestMode, bestGroups, n);
        return { squareness: bestS, mode: bestMode, groups: bestGroups, layout };
    }

    buildTiles(W, H, mode, groups, n) {
        const tiles = [];
        if (mode === 'rows') {
            let yOff = 0;
            for (const m of groups) {
                const rowH = H * m / n;
                const tileW = W / m;
                for (let i = 0; i < m; i++)
                    tiles.push({ x: i * tileW, y: yOff, w: tileW, h: rowH });
                yOff += rowH;
            }
        } else {
            let xOff = 0;
            for (const m of groups) {
                const colW = W * m / n;
                const tileH = H / m;
                for (let i = 0; i < m; i++)
                    tiles.push({ x: xOff, y: i * tileH, w: colW, h: tileH });
                xOff += colW;
            }
        }
        return tiles;
    }

    computeLayout() {
        if (!this.rootCluster) {
            console.error("No root cluster (merged) found!");
            return;
        }

        console.log("\n=== COMPUTING SQUARENESS LAYOUT ===");

        const visited = new Set();
        const buildTree = (cluster, depth = 0) => {
            if (!cluster || visited.has(cluster.path)) return null;
            visited.add(cluster.path);
            const node = { cluster, depth, children: [] };
            this.treeNodes.push(node);
            for (const child of (cluster.children || [])) {
                const cn = buildTree(child, depth + 1);
                if (cn) node.children.push(cn);
            }
            return node;
        };
        const rootNode = buildTree(this.rootCluster);

        const leaves = this.treeNodes.filter(n => n.children.length === 0);
        const leafCount = leaves.length;
        const radii = Array.from(this.clusters.values())
            .filter(c => c.radius > 0).map(c => c.radius);
        const maxRadius = radii.length > 0 ? Math.max(...radii) : 1;

        const tileSize = maxRadius * 2.2;
        const totalArea = tileSize * tileSize * leafCount * 1.1;
        const aspect = 16 / 9;
        const ROOT_H = Math.sqrt(totalArea / aspect);
        const ROOT_W = ROOT_H * aspect;

        console.log(`Leaves: ${leafCount}, maxRadius: ${maxRadius.toFixed(1)}`);
        console.log(`Root rect: ${ROOT_W.toFixed(0)} x ${ROOT_H.toFixed(0)}`);

        const rootRect = { x: -ROOT_W / 2, y: -ROOT_H / 2, w: ROOT_W, h: ROOT_H };
        this.assignLeafTiles(rootNode, rootRect);
        this.computeMergePositions(rootNode);

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

        for (const node of this.treeNodes) {
            const c = node.cluster;
            const isLeaf = node.children.length === 0;

            if (isLeaf && c.rect) {
                const cx = c.rect.x + c.rect.w / 2;
                const cy = c.rect.y + c.rect.h / 2;
                c.hierarchyPosition = new THREE.Vector3(cx, cy, 0);
                c.group.position.copy(c.hierarchyPosition);

                if (c.radius > 0) {
                    const fitDim = Math.min(c.rect.w, c.rect.h) * 0.85;
                    c.fitScale = fitDim / (2 * c.radius);
                    c.group.scale.setScalar(c.fitScale);
                }

                minX = Math.min(minX, c.rect.x);
                maxX = Math.max(maxX, c.rect.x + c.rect.w);
                minY = Math.min(minY, c.rect.y);
                maxY = Math.max(maxY, c.rect.y + c.rect.h);

            } else if (!isLeaf) {
                const pos = c.mergeTargetPosition;
                const reg = c.mergeRegion;
                if (pos) {
                    c.hierarchyPosition = pos.clone();
                    c.group.position.copy(c.hierarchyPosition);

                    if (c.radius > 0 && reg) {
                        const fitDim = Math.min(reg.w, reg.h) * 0.85;
                        c.fitScale = fitDim / (2 * c.radius);
                        c.group.scale.setScalar(c.fitScale);
                    }
                }
            }
        }

        if (minX === Infinity) {
            this.bounds = { minX: -10, maxX: 10, minY: -10, maxY: 10, width: 60, height: 60 };
        } else {
            this.bounds = {
                minX, maxX, minY, maxY,
                width: maxX - minX + 40,
                height: maxY - minY + 40
            };
        }

        for (const [path] of this.clusters) {
            if (!visited.has(path)) this.clusters.get(path).group.visible = false;
        }

        console.log(`Bounds: ${this.bounds.width.toFixed(0)} x ${this.bounds.height.toFixed(0)}`);
        console.log("=== SQUARENESS LAYOUT COMPLETE ===\n");
    }

    countLeaves(node) {
        if (!node) return 0;
        if (node.children.length === 0) return 1;
        let sum = 0;
        for (const child of node.children) sum += this.countLeaves(child);
        return sum;
    }

    assignLeafTiles(node, rect) {
        if (!node) return;
        if (node.children.length === 0) {
            node.cluster.rect = { ...rect };
            return;
        }

        const children = node.children;
        const n = children.length;
        const weights = children.map(c => this.countLeaves(c));
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        const pad = Math.min(rect.w, rect.h) * this.PADDING_FRAC;

        // Generate all possible partitions of children into contiguous groups.
        // For n children, there are 2^(n-1) ways to split (each gap is break or not).
        const partitions = [];
        const numSplits = 1 << (n - 1);
        for (let mask = 0; mask < numSplits; mask++) {
            const groups = [];
            let start = 0;
            for (let bit = 0; bit < n - 1; bit++) {
                if (mask & (1 << bit)) {
                    groups.push({ from: start, to: bit + 1 });
                    start = bit + 1;
                }
            }
            groups.push({ from: start, to: n });
            partitions.push(groups);
        }

        // Evaluate each partition in both row and column orientations.
        let bestScore = -1;
        let bestPartition = null;
        let bestOrientation = 'rows';

        for (const groups of partitions) {
            for (const orientation of ['rows', 'cols']) {
                const primaryDim = orientation === 'rows' ? rect.h : rect.w;
                const crossDim = orientation === 'rows' ? rect.w : rect.h;
                let worst = 1;

                for (const g of groups) {
                    let groupWeight = 0;
                    for (let i = g.from; i < g.to; i++) groupWeight += weights[i];
                    const groupPrimary = primaryDim * (groupWeight / totalWeight);

                    for (let i = g.from; i < g.to; i++) {
                        const childCross = crossDim * (weights[i] / groupWeight);
                        const w = (orientation === 'rows' ? childCross : groupPrimary) - pad;
                        const h = (orientation === 'rows' ? groupPrimary : childCross) - pad;
                        if (w > 0 && h > 0) {
                            worst = Math.min(worst, Math.min(w, h) / Math.max(w, h));
                        } else {
                            worst = 0;
                        }
                    }
                }

                if (worst > bestScore) {
                    bestScore = worst;
                    bestPartition = groups;
                    bestOrientation = orientation;
                }
            }
        }

        // Apply the best partition.
        let primaryOff = 0;
        const primaryTotal = bestOrientation === 'rows' ? rect.h : rect.w;
        const crossTotal = bestOrientation === 'rows' ? rect.w : rect.h;

        for (const g of bestPartition) {
            let groupWeight = 0;
            for (let i = g.from; i < g.to; i++) groupWeight += weights[i];
            const groupPrimary = primaryTotal * (groupWeight / totalWeight);

            let crossOff = 0;
            for (let i = g.from; i < g.to; i++) {
                const childCross = crossTotal * (weights[i] / groupWeight);
                let childRect;
                if (bestOrientation === 'rows') {
                    childRect = {
                        x: rect.x + crossOff + pad / 2,
                        y: rect.y + primaryOff + pad / 2,
                        w: childCross - pad,
                        h: groupPrimary - pad
                    };
                } else {
                    childRect = {
                        x: rect.x + primaryOff + pad / 2,
                        y: rect.y + crossOff + pad / 2,
                        w: groupPrimary - pad,
                        h: childCross - pad
                    };
                }
                this.assignLeafTiles(children[i], childRect);
                crossOff += childCross;
            }
            primaryOff += groupPrimary;
        }
    }

    computeMergePositions(node) {
        if (!node || node.children.length === 0) return;

        for (const child of node.children) this.computeMergePositions(child);

        const leafRects = [];
        const gatherLeaves = (n) => {
            if (n.children.length === 0 && n.cluster.rect) {
                leafRects.push(n.cluster.rect);
            }
            for (const ch of n.children) gatherLeaves(ch);
        };
        gatherLeaves(node);

        if (leafRects.length === 0) return;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const r of leafRects) {
            minX = Math.min(minX, r.x);
            maxX = Math.max(maxX, r.x + r.w);
            minY = Math.min(minY, r.y);
            maxY = Math.max(maxY, r.y + r.h);
        }

        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;

        node.cluster.mergeTargetPosition = new THREE.Vector3(cx, cy, 0);
        node.cluster.mergeRegion = {
            x: minX, y: minY,
            w: maxX - minX,
            h: maxY - minY
        };
    }
}
