"""Generate a static web UI to visualize track reprojections per cluster.

The UI loads COLMAP text outputs (vggt / vggt_pre_ba) and lets you browse clusters,
see per-image tracks, and inspect details with zoomed pixel crops.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.server
import json
import logging
import os
import pickle
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import socketserver
from typing import Any, Dict, Iterable, List, Tuple

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.visualization.track_viz_utils import collect_reprojection_pairs


logger = logging.getLogger(__name__)

RECON_OPTIONS = ("vggt", "vggt_pre_ba")
COLMAP_REQUIRED = ("cameras.txt", "images.txt", "points3D.txt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a static tracks web viewer for cluster reconstructions.")
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root directory containing cluster results (e.g. outputs/trevi_debug/results).",
    )
    parser.add_argument(
        "--cluster_tree_pkl",
        type=str,
        default=None,
        help="Path to cluster_tree.pkl. Defaults to <results_root>/cluster_tree.pkl.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root containing images.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional images directory override. Defaults to <dataset_dir>/images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the web viewer. Defaults to <results_root>/tracks_web_viz.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for loading COLMAP reconstructions.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the output directory with a local HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use when --serve is enabled.",
    )
    return parser.parse_args()


def _node_results_dir(results_root: Path, path_str: str) -> Path:
    """Map a cluster-tree path string to the corresponding results subdirectory."""
    if path_str == "root":
        return results_root
    indices = [int(p) for p in path_str.split(".") if p]
    parts: list[str] = []
    for depth in range(1, len(indices) + 1):
        partial = indices[:depth]
        name = "C_" + "_".join(str(i) for i in partial)
        parts.append(name)
    return results_root.joinpath(*parts)


def _collect_cluster_paths(node: ClusterTree, path: Tuple[int, ...] = ()) -> List[Dict[str, Any]]:
    """Return a flat list of cluster nodes with path strings and depths."""
    path_str = "root" if len(path) == 0 else ".".join(str(p) for p in path)
    nodes = [{"path": path_str, "depth": len(path)}]
    for child_idx, child in enumerate(node.children, start=1):
        nodes.extend(_collect_cluster_paths(child, path + (child_idx,)))
    return nodes


def _has_colmap_text(dir_path: Path) -> bool:
    return all((dir_path / name).exists() for name in COLMAP_REQUIRED)


def _normalize_shape(height_width: Tuple[int, int] | None, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if height_width is None:
        return fallback
    h, w = int(height_width[0]), int(height_width[1])
    if h <= 0 or w <= 0:
        return fallback
    return (h, w)


def _fallback_shape_from_camera(gtsfm_data: GtsfmData, image_idx: int) -> Tuple[int, int]:
    cam = gtsfm_data.get_camera(image_idx)
    if cam is None:
        return (1, 1)
    calib = cam.calibration()
    cx = getattr(calib, "px", None)
    cy = getattr(calib, "py", None)
    if callable(cx):
        cx = cx()
    if callable(cy):
        cy = cy()
    if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
        width = max(int(round(cx * 2)), 1)
        height = max(int(round(cy * 2)), 1)
        return (height, width)
    return (1, 1)


def _build_image_entries(
    gtsfm_data: GtsfmData, images_dir: Path, *, include_file_path: bool
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    num_tracks = gtsfm_data.number_tracks()
    allowed_track_indices = set(range(num_tracks))
    track_lengths: Dict[int, int] = {}
    for track_idx in range(num_tracks):
        track_lengths[track_idx] = int(gtsfm_data.get_track(track_idx).numberMeasurements())

    for image_idx in sorted(gtsfm_data.get_valid_camera_indices()):
        info = gtsfm_data.get_image_info(image_idx)
        name = info.name if info.name else f"image_{image_idx:06d}.jpg"
        file_path = (images_dir / name).resolve()
        if not file_path.exists():
            logger.warning("Missing image file for %s (expected %s)", name, file_path)
        fallback_shape = _fallback_shape_from_camera(gtsfm_data, image_idx)
        height, width = _normalize_shape(info.shape, fallback_shape)

        pairs = collect_reprojection_pairs(gtsfm_data, image_idx, allowed_track_indices)
        tracks: List[Dict[str, Any]] = []
        for track_idx, uv_measured, uv_reproj in pairs:
            reproj_error = float(((uv_measured[0] - uv_reproj[0]) ** 2 + (uv_measured[1] - uv_reproj[1]) ** 2) ** 0.5)
            tracks.append(
                {
                    "track_id": int(track_idx),
                    "meas": [float(uv_measured[0]), float(uv_measured[1])],
                    "reproj": [float(uv_reproj[0]), float(uv_reproj[1])],
                    "track_len": int(track_lengths.get(int(track_idx), 0)),
                    "reproj_err": reproj_error,
                }
            )

        entry: Dict[str, Any] = {
            "image_id": int(image_idx),
            "name": name,
            "width": int(width),
            "height": int(height),
            "tracks": tracks,
        }
        if include_file_path:
            entry["file_path"] = str(file_path)
        entries.append(entry)
    return entries


def _load_recon_images(recon_dir: Path, images_dir: Path, cluster_path: str, recon: str) -> List[Dict[str, Any]]:
    if not _has_colmap_text(recon_dir):
        return []
    logger.info("Loading COLMAP text for %s (%s)", cluster_path, recon)
    try:
        gtsfm_data = GtsfmData.read_colmap(str(recon_dir))
    except Exception as exc:
        logger.exception("Skipping %s/%s due to error: %s", cluster_path, recon, exc)
        return []
    return _build_image_entries(gtsfm_data, images_dir, include_file_path=True)


def _build_data_payload(
    cluster_nodes: Iterable[Dict[str, Any]], results_root: Path, images_dir: Path, num_workers: int
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    images_by_cluster: Dict[str, Dict[str, Any]] = {}

    tasks: List[Tuple[str, str, Path]] = []
    for node in cluster_nodes:
        path_str = node["path"]
        node_dir = _node_results_dir(results_root, path_str)
        label = "root" if path_str == "root" else f"C_{path_str.replace('.', '_')}"
        cluster_entry = {
            "id": path_str,
            "path": path_str,
            "label": label,
            "depth": node["depth"],
        }
        clusters.append(cluster_entry)
        images_by_cluster[path_str] = {}
        for recon in RECON_OPTIONS:
            tasks.append((path_str, recon, node_dir / recon))

    max_workers = max(1, num_workers)
    logger.info("Loading reconstructions with %d worker threads.", max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_load_recon_images, recon_dir, images_dir, path_str, recon): (path_str, recon)
            for path_str, recon, recon_dir in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            path_str, recon = future_map[future]
            images = future.result()
            if images:
                images_by_cluster[path_str][recon] = images

    return {
        "clusters": clusters,
        "imagesByCluster": images_by_cluster,
        "defaultRecon": "vggt",
        "baseImageUrl": None,
    }


def _write_html(output_dir: Path, title: str, *, serve_mode: bool) -> None:
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>__TITLE__</title>
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #0d1117;
        --bg-soft: #0f141c;
        --panel: #141923;
        --panel-strong: #181f2c;
        --text: #e6e6e6;
        --muted: #9aa4b2;
        --accent: #7aa2ff;
        --border: #263041;
        --highlight: #ffd166;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #101826 0%, #0b0f16 45%, #07090d 100%);
        color: var(--text);
      }}
      header {{
        position: sticky;
        top: 0;
        z-index: 5;
        display: flex;
        flex-wrap: wrap;
        gap: 12px 18px;
        align-items: center;
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(20, 25, 35, 0.95);
        backdrop-filter: blur(8px);
      }}
      header h1 {{
        font-size: 16px;
        font-weight: 600;
        margin: 0 12px 0 0;
      }}
      select, label, input {{
        font-size: 13px;
        color: var(--text);
      }}
      select, input[type="text"] {{
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 8px;
      }}
      input[type="range"] {{
        accent-color: var(--accent);
      }}
      .layout {{
        display: grid;
        grid-template-columns: 1.15fr 1.85fr;
        height: calc(100vh - 64px);
      }}
      .left {{
        border-right: 1px solid var(--border);
        overflow: auto;
        padding: 16px;
        background: var(--bg-soft);
      }}
      .right {{
        display: grid;
        grid-template-rows: auto 1fr auto;
        gap: 12px;
        padding: 16px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
      }}
      .sidebar-header {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 12px;
      }}
      .thumb-card {{
        display: grid;
        grid-template-columns: 220px 1fr;
        gap: 12px;
        padding: 12px;
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 12px;
        background: var(--panel);
        cursor: pointer;
        transition: border-color 0.15s ease, transform 0.15s ease;
      }}
      .thumb-card:hover {{
        border-color: var(--accent);
        transform: translateY(-1px);
      }}
      .thumb-card.selected {{
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent) inset;
      }}
      canvas {{
        background: #0b0d12;
        border-radius: 8px;
        max-width: 100%;
      }}
      .meta {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .muted {{
        color: var(--muted);
      }}
      .controls {{
        display: flex;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .zoom-pane {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 10px;
        align-items: center;
      }}
      .zoom-canvas {{
        width: 240px;
        height: 240px;
        border: 1px solid var(--border);
      }}
      .status {{
        font-size: 13px;
        color: var(--muted);
      }}
      .track-pill {{
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-size: 12px;
        display: inline-block;
      }}
      .kpi {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
      }}
      .kpi span {{
        background: #1a2332;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 2px 8px;
      }}
      .button-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      button {{
        background: #1a2332;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 10px;
        cursor: pointer;
      }}
      button.active {{
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent) inset;
      }}
      .legend-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 10px;
      }}
      .dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
      }}
      .dot-reproj {{
        background: #f25f5c;
      }}
      .dot-meas {{
        border: 2px solid #00ff80;
        background: transparent;
      }}
      .track-patches-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 10px;
        max-height: 55vh;
        overflow: auto;
      }}
      .track-patch-card {{
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px;
        background: #101723;
      }}
      .track-patch-canvas {{
        width: 100%;
        aspect-ratio: 1 / 1;
        border: 1px solid var(--border);
      }}
    </style>
  </head>
  <body>
    <header>
      <h1>Tracks Viewer</h1>
      <label>
        Cluster
        <select id="clusterSelect"></select>
      </label>
      <label>
        Recon
        <select id="reconSelect">
          <option value="vggt">vggt</option>
          <option value="vggt_pre_ba">vggt_pre_ba</option>
        </select>
      </label>
      <label>
        <input type="checkbox" id="lineToggle" checked />
        Show reprojection line
      </label>
      <label>
        Crop size
        <input type="range" id="cropSize" min="30" max="60" value="40" />
      </label>
      <label>
        Patch size (YxY)
        <input type="number" id="patchSizeInput" min="16" max="2048" step="16" value="128" style="width:90px" />
      </label>
      <label>
        Max tracks / patch
        <input type="number" id="maxPerPatchInput" min="1" max="100" step="1" value="1" style="width:70px" />
      </label>
      <span class="status" id="statusText"></span>
    </header>
    <div class="layout">
      <div class="left">
        <div class="sidebar-header">
          <input type="text" id="imageFilter" placeholder="Filter images by name or id..." />
          <div class="kpi">
            <span id="imageCount">0 images</span>
            <span id="trackCount">0 tracks</span>
          </div>
        </div>
        <div id="thumbList"></div>
      </div>
      <div class="right">
        <div class="panel controls">
          <span class="track-pill" id="trackBadge">Track: none</span>
          <span class="muted" id="imageBadge"></span>
          <span class="muted">Click any track in thumbnail or main image to highlight everywhere.</span>
          <div class="legend-chip"><span class="dot dot-reproj"></span><span class="muted">solid dot = reprojection</span></div>
          <div class="legend-chip"><span class="dot dot-meas"></span><span class="muted">green circle = measured point</span></div>
        </div>
        <div class="panel">
          <div class="button-row">
            <button id="modeAllBtn" class="active">Show all tracks</button>
            <button id="modeBestBtn">Best / patch</button>
            <button id="modeWorstBtn">Worst / patch</button>
            <button id="modeRandomBtn">Random / patch</button>
          </div>
        </div>
        <div class="panel">
          <canvas id="mainCanvas"></canvas>
        </div>
        <div class="panel zoom-pane">
          <canvas id="zoomCanvas" class="zoom-canvas" width="240" height="240"></canvas>
          <div class="muted">Hover the main image to inspect pixels.</div>
        </div>
        <div class="panel">
          <div class="muted" id="patchesInfo">Select a track to view patches across images.</div>
          <div id="trackPatchesGrid" class="track-patches-grid"></div>
        </div>
      </div>
    </div>
    <script>
      const serverMode = __SERVER_MODE__;
    </script>
    <script src="data.js"></script>
    <script>
      const data = window.TRACKS_DATA || { clusters: [], imagesByCluster: {}, defaultRecon: "vggt", baseImageUrl: null };
      const clusterSelect = document.getElementById("clusterSelect");
      const reconSelect = document.getElementById("reconSelect");
      const thumbList = document.getElementById("thumbList");
      const statusText = document.getElementById("statusText");
      const lineToggle = document.getElementById("lineToggle");
      const cropSize = document.getElementById("cropSize");
      const mainCanvas = document.getElementById("mainCanvas");
      const zoomCanvas = document.getElementById("zoomCanvas");
      const trackBadge = document.getElementById("trackBadge");
      const imageBadge = document.getElementById("imageBadge");
      const imageFilter = document.getElementById("imageFilter");
      const imageCount = document.getElementById("imageCount");
      const trackCount = document.getElementById("trackCount");
      const patchSizeInput = document.getElementById("patchSizeInput");
      const maxPerPatchInput = document.getElementById("maxPerPatchInput");
      const modeAllBtn = document.getElementById("modeAllBtn");
      const modeBestBtn = document.getElementById("modeBestBtn");
      const modeWorstBtn = document.getElementById("modeWorstBtn");
      const modeRandomBtn = document.getElementById("modeRandomBtn");
      const patchesInfo = document.getElementById("patchesInfo");
      const trackPatchesGrid = document.getElementById("trackPatchesGrid");
      const baseMainCanvas = document.createElement("canvas");
      const baseMainCtx = baseMainCanvas.getContext("2d");
      const imageCache = new Map();

      const state = {
        clusterId: null,
        recon: data.defaultRecon || "vggt",
        selectedImageId: null,
        selectedTrackId: null,
        showLine: true,
        cropSize: Number(cropSize.value),
        filterText: "",
        patchSize: Number(patchSizeInput.value),
        maxPerPatch: Number(maxPerPatchInput.value),
        patchMode: "all",
        shownTrackIds: null,
        loadingImages: false,
      };

      function hsvToRgb(h, s, v) {
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);
        let r, g, b;
        switch (i % 6) {
          case 0: r = v; g = t; b = p; break;
          case 1: r = q; g = v; b = p; break;
          case 2: r = p; g = v; b = t; break;
          case 3: r = p; g = q; b = v; break;
          case 4: r = t; g = p; b = v; break;
          case 5: r = v; g = p; b = q; break;
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
      }

      function trackColor(trackId) {
        const hue = (trackId * 0.61803398875) % 1.0;
        return hsvToRgb(hue, 0.7, 0.95);
      }

      function setStatus(msg) {
        statusText.textContent = msg || "";
      }

      function applyFilter(images) {
        const term = state.filterText.trim().toLowerCase();
        if (!term) {
          return images;
        }
        return images.filter((img) => {
          const name = img.name ? img.name.toLowerCase() : "";
          return name.includes(term) || String(img.image_id).includes(term);
        });
      }

      function getClusterImages() {
        const cluster = data.imagesByCluster[state.clusterId] || {};
        return cluster[state.recon] || [];
      }

      function getSelectedImageData(images) {
        if (!images || !images.length || state.selectedImageId === null) {
          return null;
        }
        return images.find((img) => img.image_id === state.selectedImageId) || null;
      }

      function setPatchMode(mode) {
        state.patchMode = mode;
        modeAllBtn.classList.toggle("active", mode === "all");
        modeBestBtn.classList.toggle("active", mode === "best");
        modeWorstBtn.classList.toggle("active", mode === "worst");
        modeRandomBtn.classList.toggle("active", mode === "random");
      }

      function computePatchSelection(imageData, mode) {
        if (!imageData || mode === "all") {
          return null;
        }
        const patchSize = Math.max(1, Number(state.patchSize) || 128);
        const maxPerPatch = Math.max(1, Number(state.maxPerPatch) || 1);
        const buckets = new Map();
        imageData.tracks.forEach((track) => {
          const py = Math.floor(track.meas[1] / patchSize);
          const px = Math.floor(track.meas[0] / patchSize);
          const key = `${py}_${px}`;
          if (!buckets.has(key)) {
            buckets.set(key, []);
          }
          buckets.get(key).push(track);
        });

        const chosen = new Set();
        buckets.forEach((tracks) => {
          let ordered = tracks.slice();
          if (mode === "random") {
            for (let i = ordered.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              const tmp = ordered[i];
              ordered[i] = ordered[j];
              ordered[j] = tmp;
            }
          } else if (mode === "best") {
            ordered.sort((a, b) => {
              if (b.track_len !== a.track_len) return b.track_len - a.track_len;
              return a.reproj_err - b.reproj_err;
            });
          } else if (mode === "worst") {
            ordered.sort((a, b) => {
              if (b.track_len !== a.track_len) return b.track_len - a.track_len;
              return b.reproj_err - a.reproj_err;
            });
          }
          ordered.slice(0, maxPerPatch).forEach((t) => chosen.add(t.track_id));
        });
        if (state.selectedTrackId !== null) {
          chosen.add(state.selectedTrackId);
        }
        return chosen;
      }

      async function fetchClustersIfNeeded() {
        if (!serverMode || data.clusters.length) return;
        const resp = await fetch("/api/clusters");
        const payload = await resp.json();
        data.clusters = payload.clusters || [];
        data.defaultRecon = payload.defaultRecon || "vggt";
        data.baseImageUrl = payload.baseImageUrl || null;
      }

      async function fetchImagesIfNeeded() {
        if (!serverMode || !state.clusterId) return;
        const cluster = data.imagesByCluster[state.clusterId] || {};
        if (cluster[state.recon]) return;
        state.loadingImages = true;
        setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
        const resp = await fetch(
          `/api/images?cluster=${encodeURIComponent(state.clusterId)}&recon=${encodeURIComponent(state.recon)}`
        );
        const payload = await resp.json();
        data.imagesByCluster[state.clusterId] = data.imagesByCluster[state.clusterId] || {};
        data.imagesByCluster[state.clusterId][state.recon] = payload.images || [];
        state.loadingImages = false;
      }

      function populateClusters() {
        const prevClusterId = state.clusterId;
        clusterSelect.innerHTML = "";
        data.clusters.forEach((cluster) => {
          const option = document.createElement("option");
          option.value = cluster.id;
          option.textContent = `${cluster.label}`;
          clusterSelect.appendChild(option);
        });
        const hasPrev = prevClusterId && data.clusters.some((cluster) => cluster.id === prevClusterId);
        state.clusterId = hasPrev ? prevClusterId : (data.clusters.length ? data.clusters[0].id : null);
        clusterSelect.value = state.clusterId;
      }

      function drawTracks(ctx, tracks, scaleX, scaleY, selectedTrackId, showLine, shownTrackIds) {
        tracks.forEach((track) => {
          if (shownTrackIds && !shownTrackIds.has(track.track_id) && track.track_id !== selectedTrackId) {
            return;
          }
          const [r, g, b] = trackColor(track.track_id);
          const isSelected = selectedTrackId !== null && track.track_id === selectedTrackId;
          const isDimmed = selectedTrackId !== null && !isSelected;
          const lineWidth = isSelected ? 2.5 : 1.2;
          const dotRadius = isSelected ? 4 : 2.2;
          const measX = track.meas[0] * scaleX;
          const measY = track.meas[1] * scaleY;
          const repX = track.reproj[0] * scaleX;
          const repY = track.reproj[1] * scaleY;

          if (showLine) {
            ctx.strokeStyle = isSelected
              ? "rgba(255, 209, 102, 0.95)"
              : (isDimmed ? "rgba(255,255,255,0.08)" : "rgba(255, 255, 255, 0.35)");
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(measX, measY);
            ctx.lineTo(repX, repY);
            ctx.stroke();
          }

          const fillAlpha = isDimmed ? 0.18 : 0.95;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${fillAlpha})`;
          ctx.beginPath();
          ctx.arc(repX, repY, dotRadius, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = isDimmed ? "rgba(0, 255, 128, 0.25)" : "rgba(0, 255, 128, 0.9)";
          ctx.lineWidth = Math.max(1, lineWidth);
          ctx.beginPath();
          ctx.arc(measX, measY, dotRadius, 0, Math.PI * 2);
          ctx.stroke();

          if (isSelected) {
            ctx.strokeStyle = "rgba(255, 209, 102, 1.0)";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(measX, measY, dotRadius + 3, 0, Math.PI * 2);
            ctx.stroke();
          }
        });
      }

      function imageSrc(imageData) {
        if (data.baseImageUrl) {
          return `${data.baseImageUrl}?name=${encodeURIComponent(imageData.name)}`;
        }
        return imageData.file_path;
      }

      function loadImageCached(src) {
        if (imageCache.has(src)) {
          return imageCache.get(src);
        }
        const promise = new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
          img.src = src;
        });
        imageCache.set(src, promise);
        return promise;
      }

      function drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY) {
        // Measured point: green cross.
        ctx.strokeStyle = "rgba(0, 255, 128, 1.0)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(measX - 4, measY);
        ctx.lineTo(measX + 4, measY);
        ctx.moveTo(measX, measY - 4);
        ctx.lineTo(measX, measY + 4);
        ctx.stroke();
        // Reprojected point: yellow cross.
        ctx.strokeStyle = "rgba(255, 220, 120, 1.0)";
        ctx.beginPath();
        ctx.moveTo(reprojX - 4, reprojY);
        ctx.lineTo(reprojX + 4, reprojY);
        ctx.moveTo(reprojX, reprojY - 4);
        ctx.lineTo(reprojX, reprojY + 4);
        ctx.stroke();
      }

      function renderThumbnail(imageData, container, onSelect) {
        const card = document.createElement("div");
        card.className = "thumb-card";
        if (imageData.image_id === state.selectedImageId) {
          card.classList.add("selected");
        }
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const targetW = 220;
        const scale = targetW / imageData.width;
        canvas.width = targetW;
        canvas.height = Math.round(imageData.height * scale);

        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawTracks(
            ctx,
            imageData.tracks,
            scale,
            scale,
            state.selectedTrackId,
            state.showLine,
            state.shownTrackIds
          );
        };
        img.src = imageSrc(imageData);

        card.addEventListener("click", () => onSelect(imageData));
        canvas.addEventListener("click", (evt) => {
          evt.stopPropagation();
          const rect = canvas.getBoundingClientRect();
          const x = evt.clientX - rect.left;
          const y = evt.clientY - rect.top;
          const trackId = pickTrack(imageData, x / scale, y / scale, state.shownTrackIds);
          if (trackId !== null) {
            state.selectedTrackId = trackId;
            const t = imageData.tracks.find((tr) => tr.track_id === trackId);
            if (t) {
              trackBadge.textContent = `Track: ${trackId} | len=${t.track_len} | err=${t.reproj_err.toFixed(2)}px`;
            } else {
              trackBadge.textContent = `Track: ${trackId}`;
            }
            renderAll();
          }
        });

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `<div><strong>${imageData.name}</strong></div>
          <div class="muted">ID: ${imageData.image_id}</div>
          <div class="muted">Tracks: ${imageData.tracks.length}</div>`;
        card.appendChild(canvas);
        card.appendChild(meta);
        container.appendChild(card);
      }

      function pickTrack(imageData, x, y, shownTrackIds) {
        let best = null;
        let bestDist = Infinity;
        const radius = 10;
        imageData.tracks.forEach((track) => {
          if (shownTrackIds && !shownTrackIds.has(track.track_id)) {
            return;
          }
          const dx = track.meas[0] - x;
          const dy = track.meas[1] - y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < radius && dist < bestDist) {
            bestDist = dist;
            best = track.track_id;
          }
        });
        return best;
      }

      function renderMainImage(imageData) {
        if (!imageData) {
          mainCanvas.width = 10;
          mainCanvas.height = 10;
          baseMainCanvas.width = 10;
          baseMainCanvas.height = 10;
          const ctx = mainCanvas.getContext("2d");
          ctx.clearRect(0, 0, 10, 10);
          baseMainCtx.clearRect(0, 0, 10, 10);
          return;
        }
        mainCanvas.width = imageData.width;
        mainCanvas.height = imageData.height;
        baseMainCanvas.width = imageData.width;
        baseMainCanvas.height = imageData.height;
        const ctx = mainCanvas.getContext("2d");
        const img = new Image();
        img.onload = () => {
          baseMainCtx.clearRect(0, 0, imageData.width, imageData.height);
          baseMainCtx.drawImage(img, 0, 0, imageData.width, imageData.height);
          ctx.drawImage(img, 0, 0, imageData.width, imageData.height);
          drawTracks(
            ctx,
            imageData.tracks,
            1,
            1,
            state.selectedTrackId,
            state.showLine,
            state.shownTrackIds
          );
        };
        img.src = imageSrc(imageData);
        imageBadge.textContent = `${imageData.name} (${imageData.width}x${imageData.height})`;
      }

      async function renderSelectedTrackPatches(images) {
        trackPatchesGrid.innerHTML = "";
        if (state.selectedTrackId === null) {
          patchesInfo.textContent = "Select a track to view patches across images.";
          return;
        }
        const patchImages = images
          .map((img) => {
            const t = img.tracks.find((tr) => tr.track_id === state.selectedTrackId);
            return t ? { image: img, track: t } : null;
          })
          .filter((v) => v !== null);
        patchesInfo.textContent = `Track ${state.selectedTrackId} appears in ${patchImages.length} image(s).`;
        if (!patchImages.length) {
          return;
        }
        const patchSize = Math.max(30, Number(state.cropSize) || 40);
        const outSize = 180;
        for (const item of patchImages) {
          const card = document.createElement("div");
          card.className = "track-patch-card";
          const label = document.createElement("div");
          label.className = "muted";
          label.textContent = `${item.image.name} (ID ${item.image.image_id})`;
          const canvas = document.createElement("canvas");
          canvas.className = "track-patch-canvas";
          canvas.width = outSize;
          canvas.height = outSize;
          card.appendChild(label);
          card.appendChild(canvas);
          trackPatchesGrid.appendChild(card);

          try {
            const img = await loadImageCached(imageSrc(item.image));
            const ctx = canvas.getContext("2d");
            const cx = Math.round(item.track.meas[0]);
            const cy = Math.round(item.track.meas[1]);
            const half = Math.floor(patchSize / 2);
            const sx = Math.max(0, cx - half);
            const sy = Math.max(0, cy - half);
            const sw = Math.min(patchSize, item.image.width - sx);
            const sh = Math.min(patchSize, item.image.height - sy);
            ctx.clearRect(0, 0, outSize, outSize);
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, sx, sy, sw, sh, 0, 0, outSize, outSize);
            const measX = ((item.track.meas[0] - sx) / sw) * outSize;
            const measY = ((item.track.meas[1] - sy) / sh) * outSize;
            const reprojX = ((item.track.reproj[0] - sx) / sw) * outSize;
            const reprojY = ((item.track.reproj[1] - sy) / sh) * outSize;
            drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY);
          } catch (_) {
            // Keep card with label if image failed to load.
          }
        }
      }

      async function renderAll() {
        if (serverMode) {
          await fetchClustersIfNeeded();
          populateClusters();
          await fetchImagesIfNeeded();
        }
        const images = getClusterImages();
        const filtered = applyFilter(images);
        thumbList.innerHTML = "";
        if (!state.clusterId) {
          setStatus("No clusters available.");
          imageCount.textContent = "0 images";
          trackCount.textContent = "0 tracks";
          return;
        }
        if (!images.length) {
          setStatus("No COLMAP data for this cluster/recon.");
          imageCount.textContent = "0 images";
          trackCount.textContent = "0 tracks";
          renderMainImage(null);
          return;
        }
        setStatus(`${filtered.length} / ${images.length} images`);
        const trackTotal = filtered.reduce((acc, img) => acc + img.tracks.length, 0);
        imageCount.textContent = `${filtered.length} images`;
        trackCount.textContent = `${trackTotal} tracks`;
        if (state.selectedImageId === null || !filtered.find((i) => i.image_id === state.selectedImageId)) {
          state.selectedImageId = filtered.length ? filtered[0].image_id : null;
        }
        const selected = filtered.find((i) => i.image_id === state.selectedImageId) || null;
        state.shownTrackIds = computePatchSelection(selected, state.patchMode);
        const shownTrackCount = state.shownTrackIds ? state.shownTrackIds.size : trackTotal;
        if (state.patchMode !== "all") {
          setStatus(
            `${filtered.length} / ${images.length} images, showing ${shownTrackCount} tracks via ${state.patchMode} mode`
          );
        }
        filtered.forEach((img) => {
          renderThumbnail(img, thumbList, (imageData) => {
            state.selectedImageId = imageData.image_id;
            renderAll();
          });
        });
        renderMainImage(selected);
        await renderSelectedTrackPatches(images);
      }

      function setupZoom() {
        const ctx = zoomCanvas.getContext("2d");
        mainCanvas.addEventListener("mousemove", (evt) => {
          const images = applyFilter(getClusterImages());
          const selectedImage = getSelectedImageData(images);
          const rect = mainCanvas.getBoundingClientRect();
          const x = evt.clientX - rect.left;
          const y = evt.clientY - rect.top;
          const scaleX = mainCanvas.width / rect.width;
          const scaleY = mainCanvas.height / rect.height;
          const cx = Math.round(x * scaleX);
          const cy = Math.round(y * scaleY);
          const size = state.cropSize;
          const half = Math.floor(size / 2);
          const sx = Math.max(0, cx - half);
          const sy = Math.max(0, cy - half);
          const sw = Math.min(size, mainCanvas.width - sx);
          const sh = Math.min(size, mainCanvas.height - sy);
          ctx.clearRect(0, 0, zoomCanvas.width, zoomCanvas.height);
          ctx.imageSmoothingEnabled = false;
          // Use clean source image for zoom (no overlay clutter), then draw only tiny selected marker.
          ctx.drawImage(baseMainCanvas, sx, sy, sw, sh, 0, 0, zoomCanvas.width, zoomCanvas.height);

          if (selectedImage && state.selectedTrackId !== null) {
            const selectedTrack = selectedImage.tracks.find((t) => t.track_id === state.selectedTrackId);
            if (selectedTrack) {
              const measIn = selectedTrack.meas[0] >= sx && selectedTrack.meas[0] <= (sx + sw)
                && selectedTrack.meas[1] >= sy && selectedTrack.meas[1] <= (sy + sh);
              const reprojIn = selectedTrack.reproj[0] >= sx && selectedTrack.reproj[0] <= (sx + sw)
                && selectedTrack.reproj[1] >= sy && selectedTrack.reproj[1] <= (sy + sh);
              if (measIn || reprojIn) {
                const measX = ((selectedTrack.meas[0] - sx) / sw) * zoomCanvas.width;
                const measY = ((selectedTrack.meas[1] - sy) / sh) * zoomCanvas.height;
                const reprojX = ((selectedTrack.reproj[0] - sx) / sw) * zoomCanvas.width;
                const reprojY = ((selectedTrack.reproj[1] - sy) / sh) * zoomCanvas.height;
                drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY);
              }
            }
          }
        });
      }

      function setupMainCanvasSelection() {
        mainCanvas.addEventListener("click", (evt) => {
          const images = applyFilter(getClusterImages());
          const selected = getSelectedImageData(images);
          if (!selected) {
            return;
          }
          const rect = mainCanvas.getBoundingClientRect();
          const x = (evt.clientX - rect.left) * (mainCanvas.width / rect.width);
          const y = (evt.clientY - rect.top) * (mainCanvas.height / rect.height);
          const trackId = pickTrack(selected, x, y, state.shownTrackIds);
          if (trackId === null) {
            return;
          }
          state.selectedTrackId = trackId;
          const t = selected.tracks.find((tr) => tr.track_id === trackId);
          if (t) {
            trackBadge.textContent = `Track: ${trackId} | len=${t.track_len} | err=${t.reproj_err.toFixed(2)}px`;
          } else {
            trackBadge.textContent = `Track: ${trackId}`;
          }
          renderAll();
        });
      }

      async function init() {
        await fetchClustersIfNeeded();
        populateClusters();
        reconSelect.value = state.recon;
        clusterSelect.addEventListener("change", () => {
          state.clusterId = clusterSelect.value;
          state.selectedImageId = null;
          if (serverMode) {
            setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
          }
          renderAll();
        });
        reconSelect.addEventListener("change", () => {
          state.recon = reconSelect.value;
          state.selectedImageId = null;
          if (serverMode) {
            setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
          }
          renderAll();
        });
        lineToggle.addEventListener("change", () => {
          state.showLine = lineToggle.checked;
          renderAll();
        });
        cropSize.addEventListener("input", () => {
          state.cropSize = Number(cropSize.value);
        });
        imageFilter.addEventListener("input", () => {
          state.filterText = imageFilter.value;
          state.selectedImageId = null;
          renderAll();
        });
        patchSizeInput.addEventListener("change", () => {
          state.patchSize = Math.max(16, Number(patchSizeInput.value) || 128);
          renderAll();
        });
        maxPerPatchInput.addEventListener("change", () => {
          state.maxPerPatch = Math.max(1, Number(maxPerPatchInput.value) || 1);
          renderAll();
        });
        modeAllBtn.addEventListener("click", () => {
          setPatchMode("all");
          renderAll();
        });
        modeBestBtn.addEventListener("click", () => {
          setPatchMode("best");
          renderAll();
        });
        modeWorstBtn.addEventListener("click", () => {
          setPatchMode("worst");
          renderAll();
        });
        modeRandomBtn.addEventListener("click", () => {
          setPatchMode("random");
          renderAll();
        });
        document.addEventListener("keydown", (evt) => {
          if (evt.key !== "Escape") {
            return;
          }
          state.selectedTrackId = null;
          trackBadge.textContent = "Track: none";
          renderAll();
        });
        setupZoom();
        setupMainCanvasSelection();
        renderAll();
      }

      init();
    </script>
  </body>
</html>
"""
    html = html.replace("__TITLE__", title)
    html = html.replace("__SERVER_MODE__", "true" if serve_mode else "false")
    # The template originally used doubled braces for f-string safety; normalize CSS braces now.
    style_start = html.find("<style>")
    style_end = html.find("</style>", style_start)
    if style_start != -1 and style_end != -1:
        css_block = html[style_start:style_end]
        css_block = css_block.replace("{{", "{").replace("}}", "}")
        html = html[:style_start] + css_block + html[style_end:]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    results_root = Path(args.results_root).resolve()
    cluster_tree_pkl = (
        Path(args.cluster_tree_pkl).resolve()
        if args.cluster_tree_pkl is not None
        else results_root / "cluster_tree.pkl"
    )
    if not cluster_tree_pkl.exists():
        raise FileNotFoundError(f"cluster_tree_pkl does not exist: {cluster_tree_pkl}")

    images_dir = Path(args.images_dir).resolve() if args.images_dir else Path(args.dataset_dir).resolve() / "images"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "tracks_web_viz"

    with cluster_tree_pkl.open("rb") as f:
        cluster_tree = pickle.load(f)
    if not isinstance(cluster_tree, ClusterTree):
        raise TypeError(f"Expected ClusterTree, got {type(cluster_tree)}")

    cluster_nodes = _collect_cluster_paths(cluster_tree)
    num_workers = args.num_workers or min(16, (os.cpu_count() or 4))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.serve:
        payload = _build_data_payload(cluster_nodes, results_root, images_dir, num_workers)
        data_js = "window.TRACKS_DATA = " + json.dumps(payload) + ";"
        (output_dir / "data.js").write_text(data_js, encoding="utf-8")
        _write_html(output_dir, "Tracks Viewer", serve_mode=False)
        logger.info("Wrote tracks web viewer to %s", output_dir / "index.html")
        return

    (output_dir / "data.js").write_text("window.TRACKS_DATA = null;", encoding="utf-8")
    _write_html(output_dir, "Tracks Viewer", serve_mode=True)
    logger.info("Wrote tracks web viewer to %s", output_dir / "index.html")

    cluster_entries = [
        {
            "id": node["path"],
            "path": node["path"],
            "label": "root" if node["path"] == "root" else f"C_{node['path'].replace('.', '_')}",
            "depth": node["depth"],
        }
        for node in cluster_nodes
    ]
    cluster_index = {entry["path"]: entry for entry in cluster_entries}
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    class TracksRequestHandler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any]) -> None:
            data_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data_bytes)))
            self.end_headers()
            self.wfile.write(data_bytes)

        def _send_file(self, path: Path) -> None:
            if not path.exists():
                self.send_error(404, "File not found")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            if path.suffix.lower() == ".html":
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif path.suffix.lower() == ".js":
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
            elif path.suffix.lower() in {".jpg", ".jpeg"}:
                self.send_header("Content-Type", "image/jpeg")
            elif path.suffix.lower() == ".png":
                self.send_header("Content-Type", "image/png")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in {"/", "/index.html"}:
                return self._send_file(output_dir / "index.html")
            if path == "/data.js":
                return self._send_file(output_dir / "data.js")
            if path == "/api/clusters":
                return self._send_json(
                    {
                        "clusters": cluster_entries,
                        "defaultRecon": "vggt",
                        "baseImageUrl": "/image",
                    }
                )
            if path == "/api/images":
                cluster = (query.get("cluster") or [""])[0]
                recon = (query.get("recon") or [""])[0]
                if cluster not in cluster_index or recon not in RECON_OPTIONS:
                    return self._send_json({"images": []})
                cache_key = (cluster, recon)
                if cache_key not in cache:
                    node_dir = _node_results_dir(results_root, cluster)
                    recon_dir = node_dir / recon
                    if not _has_colmap_text(recon_dir):
                        cache[cache_key] = []
                    else:
                        gtsfm_data = GtsfmData.read_colmap(str(recon_dir))
                        cache[cache_key] = _build_image_entries(
                            gtsfm_data, images_dir, include_file_path=False
                        )
                return self._send_json({"images": cache[cache_key]})
            if path == "/image":
                name = (query.get("name") or [""])[0]
                name = unquote(name)
                if not name:
                    self.send_error(404, "File not found")
                    return
                candidate = (images_dir / name).resolve()
                if not str(candidate).startswith(str(images_dir.resolve())):
                    self.send_error(403, "Forbidden")
                    return
                return self._send_file(candidate)

            return self._send_file(output_dir / path.lstrip("/"))

    with http.server.ThreadingHTTPServer(("", args.port), TracksRequestHandler) as httpd:
        logger.info("Serving %s at http://localhost:%d", output_dir, args.port)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
