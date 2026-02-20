"""Visualize a serialized ClusterTree as an interactive HTML page.

Features:
- Collapsible tree view of all cluster nodes.
- Per-node details panel with camera index lists and edge counts.
- Optional image thumbnails for cameras (loaded via a GTSFM loader config).

Examples:
    python scripts/visualize_cluster_tree.py \
        --cluster_tree_pkl /path/to/results/cluster_tree.pkl \
        --output_html /path/to/results/cluster_tree_viz.html

    python scripts/visualize_cluster_tree.py \
        --cluster_tree_pkl /path/to/results/cluster_tree.pkl \
        --loader_config olsson \
        --dataset_dir /path/to/dataset \
        --images_dir /path/to/dataset/images
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import pickle
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image as PILImage

from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import visibility_graph_keys

if TYPE_CHECKING:
    from gtsfm.loader.loader_base import LoaderBase

logger = logging.getLogger(__name__)


def _build_loader(
    loader_config: str,
    dataset_dir: str,
    images_dir: Optional[str],
    max_resolution: Optional[int],
    colmap_files_subdir: Optional[str],
) -> "LoaderBase":
    """Instantiate a loader from gtsfm/configs/loader/<loader_config>.yaml."""
    import hydra
    from hydra.utils import instantiate

    overrides: List[str] = [f"dataset_dir={dataset_dir}"]
    if images_dir is not None:
        overrides.append(f"images_dir={images_dir}")
    if max_resolution is not None:
        overrides.append(f"max_resolution={max_resolution}")
    if colmap_files_subdir is not None:
        overrides.append(f"colmap_files_subdir={colmap_files_subdir}")

    config_dir = Path(__file__).resolve().parents[1] / "gtsfm" / "configs" / "loader"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=loader_config, overrides=overrides)
    return instantiate(cfg)


def _normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 for thumbnail encoding."""
    arr = image_array
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _to_jpeg_data_uri(image_array: np.ndarray, thumb_size_px: int) -> str:
    """Encode an RGB/grayscale array as a base64 JPEG data URI."""
    arr = _normalize_to_uint8(image_array)
    if arr.ndim == 2:
        pil_img = PILImage.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 1:
        pil_img = PILImage.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        pil_img = PILImage.fromarray(arr[:, :, :3], mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape for thumbnail: {arr.shape}")

    pil_img.thumbnail((thumb_size_px, thumb_size_px))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=80, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _tree_to_dict(
    node: ClusterTree,
    node_id: str,
    path: Tuple[int, ...],
    depth: int,
    similarity_matrix: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Convert ClusterTree to a JSON-serializable tree with useful summaries."""
    local_edges = [tuple(sorted((int(i), int(j)))) for i, j in node.value]
    local_keys = sorted(visibility_graph_keys(local_edges))
    local_edge_tuples = sorted(set(local_edges))
    local_edge_records: List[List[Any]] = []
    for i, j in local_edge_tuples:
        score: Optional[float] = None
        if similarity_matrix is not None and i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
            score = float(similarity_matrix[i, j])
        local_edge_records.append([i, j, score])

    child_payloads: List[Dict[str, Any]] = []
    subtree_key_set: Set[int] = set(local_keys)
    subtree_edge_count = len(local_edge_tuples)
    subtree_edges: Set[Tuple[int, int]] = set(local_edge_tuples)

    for child_idx, child in enumerate(node.children, start=1):
        child_id = f"{node_id}.{child_idx}"
        child_payload = _tree_to_dict(
            node=child,
            node_id=child_id,
            path=path + (child_idx,),
            depth=depth + 1,
            similarity_matrix=similarity_matrix,
        )
        child_payloads.append(child_payload)
        subtree_key_set.update(child_payload["subtree_keys"])
        subtree_edge_count += int(child_payload["subtree_edge_count"])
        for i, j, _ in child_payload["subtree_edges"]:
            subtree_edges.add((int(i), int(j)))

    subtree_keys = sorted(subtree_key_set)
    subtree_edge_records: List[List[Any]] = []
    for i, j in sorted(subtree_edges):
        score: Optional[float] = None
        if similarity_matrix is not None and i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
            score = float(similarity_matrix[i, j])
        subtree_edge_records.append([i, j, score])

    return {
        "id": node_id,
        "path": "root" if len(path) == 0 else ".".join(str(p) for p in path),
        "depth": depth,
        "is_leaf": len(child_payloads) == 0,
        "num_children": len(child_payloads),
        "local_edge_count": len(local_edge_tuples),
        "subtree_edge_count": subtree_edge_count,
        "local_keys": local_keys,
        "subtree_keys": subtree_keys,
        "local_edges": local_edge_records,
        "subtree_edges": subtree_edge_records,
        "children": child_payloads,
    }


def _collect_all_camera_indices(node_dict: Dict[str, Any]) -> Set[int]:
    """Collect the union of subtree camera indices across all nodes."""
    all_keys: Set[int] = set(node_dict["subtree_keys"])
    for child in node_dict["children"]:
        all_keys.update(_collect_all_camera_indices(child))
    return all_keys


def _load_similarity_matrix(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load similarity matrix from CSV txt if available."""
    if path is None:
        return None
    if not path.exists():
        logger.warning("Similarity matrix path does not exist: %s", path)
        return None
    try:
        sim = np.loadtxt(str(path), delimiter=",")
    except Exception as exc:
        logger.warning("Failed to load similarity matrix from %s: %s", path, exc)
        return None
    if sim.ndim != 2:
        logger.warning("Similarity matrix at %s is not 2D.", path)
        return None
    logger.info("Loaded similarity matrix with shape %s from %s", sim.shape, path)
    return sim


def _build_thumbnail_payload(
    loader: "LoaderBase",
    camera_indices: Iterable[int],
    thumb_size_px: int,
) -> Dict[str, Dict[str, str]]:
    """Return payload keyed by camera index with optional name + thumbnail."""
    indices = sorted(set(camera_indices))
    filenames = loader.image_filenames()

    payload: Dict[str, Dict[str, str]] = {}
    for idx in indices:
        camera_key = str(idx)
        item: Dict[str, str] = {}
        if 0 <= idx < len(filenames):
            item["filename"] = str(filenames[idx])
        else:
            item["filename"] = f"camera_{idx}"

        try:
            img = loader.get_image(idx)
            item["thumbnail_data_uri"] = _to_jpeg_data_uri(img.value_array, thumb_size_px=thumb_size_px)
        except Exception as exc:
            item["thumbnail_error"] = str(exc)
        payload[camera_key] = item
    return payload


def _read_colmap_image_filenames(images_txt_path: Path) -> List[str]:
    """Read image filename order from COLMAP images.txt."""
    if not images_txt_path.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt_path}")

    raw_lines = images_txt_path.read_text(encoding="utf-8").splitlines()
    content_lines = [line.strip() for line in raw_lines if line.strip() and not line.lstrip().startswith("#")]
    image_lines = content_lines[0::2]

    filenames: List[str] = []
    for line in image_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        filenames.append(parts[-1])
    return filenames


def _build_colmap_thumbnail_payload_without_loader(
    camera_indices: Iterable[int],
    images_dir: Path,
    colmap_files_dir: Path,
    thumb_size_px: int,
) -> Dict[str, Dict[str, str]]:
    """Fallback when Hydra/loader dependencies are unavailable."""
    raw_image_filenames = _read_colmap_image_filenames(colmap_files_dir / "images.txt")
    # Mirror ColmapLoader behavior: preserve COLMAP ordering but drop entries
    # for missing files, so index mapping matches loader indices.
    image_filenames = [name for name in raw_image_filenames if (images_dir / name).exists()]
    indices = sorted(set(camera_indices))
    payload: Dict[str, Dict[str, str]] = {}

    for idx in indices:
        camera_key = str(idx)
        item: Dict[str, str] = {}
        if 0 <= idx < len(image_filenames):
            filename = image_filenames[idx]
        else:
            filename = f"camera_{idx}"
        item["filename"] = filename

        image_path = images_dir / filename
        try:
            with PILImage.open(image_path) as pil_img:
                arr = np.asarray(pil_img.convert("RGB"))
            item["thumbnail_data_uri"] = _to_jpeg_data_uri(arr, thumb_size_px=thumb_size_px)
        except Exception as exc:
            item["thumbnail_error"] = str(exc)
        payload[camera_key] = item
    return payload


def _sorted_image_filenames_from_dir(images_dir: Path) -> List[str]:
    """Mirror OlssonLoader filename ordering (sorted by path string)."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.PNG")
    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(images_dir.glob(pattern))
    return [p.name for p in sorted(image_paths)]


def _build_olsson_thumbnail_payload_without_loader(
    camera_indices: Iterable[int],
    dataset_dir: Path,
    images_dir: Optional[Path],
    thumb_size_px: int,
) -> Dict[str, Dict[str, str]]:
    """Fallback for Olsson loader when Hydra is unavailable."""
    resolved_images_dir = images_dir if images_dir is not None else dataset_dir / "images"
    image_filenames = _sorted_image_filenames_from_dir(resolved_images_dir)
    indices = sorted(set(camera_indices))
    payload: Dict[str, Dict[str, str]] = {}

    for idx in indices:
        camera_key = str(idx)
        item: Dict[str, str] = {}
        if 0 <= idx < len(image_filenames):
            filename = image_filenames[idx]
        else:
            filename = f"camera_{idx}"
        item["filename"] = filename

        image_path = resolved_images_dir / filename
        try:
            with PILImage.open(image_path) as pil_img:
                arr = np.asarray(pil_img.convert("RGB"))
            item["thumbnail_data_uri"] = _to_jpeg_data_uri(arr, thumb_size_px=thumb_size_px)
        except Exception as exc:
            item["thumbnail_error"] = str(exc)
        payload[camera_key] = item
    return payload


def _load_cytoscape_source(path: Path) -> str:
    """Load Cytoscape JavaScript bundle from local file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cytoscape bundle not found at {path}. Download one to scripts/assets/cytoscape.min.js."
        )
    return path.read_text(encoding="utf-8")


def _escape_js_for_inline_script(js_source: str) -> str:
    """Prevent accidental script tag termination inside inline JS."""
    return js_source.replace("</script", "<\\/script")


def _render_html(
    tree_data: Dict[str, Any],
    output_html: Path,
    thumbnail_payload: Optional[Dict[str, Dict[str, str]]],
    max_thumbnails_per_node: int,
    cytoscape_source: str,
) -> None:
    """Write a self-contained HTML with embedded tree and thumbnails JSON."""
    tree_json = json.dumps(tree_data, separators=(",", ":"))
    thumbs_json = json.dumps(thumbnail_payload or {}, separators=(",", ":"))

    cytoscape_inline = _escape_js_for_inline_script(cytoscape_source)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cluster Tree Viewer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --card: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #60a5fa;
      --border: #374151;
      --hover: #243447;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 42%) 1fr;
      gap: 10px;
      height: 100vh;
      padding: 10px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .panel-header {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .panel-title {{
      font-weight: 600;
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    button {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 9px;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{ background: #334155; }}
    input[type="text"] {{
      background: #0b1220;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      min-width: 220px;
      font-size: 12px;
    }}
    select {{
      background: #0b1220;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
    }}
    .tree-wrap {{
      overflow: auto;
      padding: 10px 12px;
      min-height: 0;
    }}
    .tree, .tree ul {{
      list-style: none;
      margin: 0;
      padding-left: 16px;
    }}
    .node-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 6px;
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .node-row:hover {{ background: var(--hover); }}
    .node-row.selected {{
      background: #1d3557;
      border-color: #3b82f6;
    }}
    .twisty {{
      width: 14px;
      text-align: center;
      color: var(--muted);
      user-select: none;
    }}
    .node-label {{
      font-size: 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .badge {{
      background: #0b1220;
      border: 1px solid var(--border);
      color: #cbd5e1;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
    }}
    .details {{
      padding: 12px;
      overflow: auto;
      min-height: 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 10px;
      padding: 10px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
    }}
    .meta-item {{
      background: #111827;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
    }}
    .meta-key {{
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .codebox {{
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      max-height: 160px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }}
    .thumb {{
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #0b1220;
    }}
    .thumb img {{
      width: 100%;
      height: 110px;
      object-fit: cover;
      display: block;
      background: #0b1220;
    }}
    .thumb-caption {{
      padding: 6px 8px;
      font-size: 11px;
      color: #d1d5db;
      word-break: break-all;
    }}
    .graph-shell {{
      position: relative;
      height: 460px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #0b1220;
      overflow: hidden;
      margin-top: 8px;
    }}
    .graph-cy {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }}
    .legend {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 6px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .legend-bar {{
      width: 110px;
      height: 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: linear-gradient(to right, #334155, #3b82f6, #22c55e);
    }}
    .graph-tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(2, 6, 23, 0.95);
      color: #e5e7eb;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 4px 6px;
      font-size: 11px;
      z-index: 10;
      display: none;
    }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">Cluster Tree</div>
        <div class="controls">
          <input id="searchInput" type="text" placeholder="Find camera index (e.g. 128)" />
          <button id="expandAllBtn">Expand all</button>
          <button id="collapseAllBtn">Collapse all</button>
        </div>
      </div>
      <div class="tree-wrap">
        <ul id="treeRoot" class="tree"></ul>
      </div>
    </section>
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">Node Details</div>
        <div class="controls">
          <label for="thumbScopeSelect" class="muted" style="font-size:12px;">Thumbnails:</label>
          <select id="thumbScopeSelect">
            <option value="subtree">Subtree cams</option>
            <option value="local">Local cams only</option>
          </select>
          <button id="toggleGraphBtn" type="button">Hide graph</button>
        </div>
      </div>
      <div id="details" class="details">
        <div class="card muted">Select a node from the tree.</div>
      </div>
    </section>
  </div>

  <script>{cytoscape_inline}</script>
  <script>
    const TREE_DATA = {tree_json};
    const THUMB_DATA = {thumbs_json};
    const MAX_THUMBS_PER_NODE = {max_thumbnails_per_node};
    const MAX_GRAPH_NODES = 80;

    const state = {{
      selectedNodeId: null,
      expanded: new Set(["root"]),
      rowById: new Map(),
      childrenUlById: new Map(),
      searchCameraIdx: null,
      thumbnailScope: "subtree",
      showSimilarityGraph: true,
      pendingGraphData: null,
      cy: null,
    }};

    function allNodeIds(node, out=[]) {{
      out.push(node.id);
      for (const c of node.children) allNodeIds(c, out);
      return out;
    }}

    function subtreeContainsCamera(node, cameraIdx) {{
      if (cameraIdx === null) return true;
      return node.subtree_keys.includes(cameraIdx);
    }}

    function createNodeElement(node) {{
      const li = document.createElement("li");
      li.dataset.nodeId = node.id;

      const row = document.createElement("div");
      row.className = "node-row";
      row.dataset.nodeId = node.id;

      const twisty = document.createElement("span");
      twisty.className = "twisty";
      twisty.textContent = node.children.length ? (state.expanded.has(node.id) ? "▾" : "▸") : "•";
      row.appendChild(twisty);

      const label = document.createElement("span");
      label.className = "node-label";
      label.innerHTML = `
        <span><strong>${{node.path}}</strong></span>
        <span class="badge">local cams: ${{node.local_keys.length}}</span>
        <span class="badge">subtree cams: ${{node.subtree_keys.length}}</span>
        <span class="badge">subtree edges: ${{node.subtree_edge_count}}</span>
      `;
      row.appendChild(label);

      row.addEventListener("click", (evt) => {{
        const target = evt.target;
        if (target === twisty && node.children.length) {{
          toggleNode(node.id);
          return;
        }}
        selectNode(node);
      }});

      li.appendChild(row);
      state.rowById.set(node.id, row);

      if (node.children.length) {{
        const ul = document.createElement("ul");
        ul.style.display = state.expanded.has(node.id) ? "block" : "none";
        for (const child of node.children) {{
          ul.appendChild(createNodeElement(child));
        }}
        state.childrenUlById.set(node.id, ul);
        li.appendChild(ul);
      }}
      return li;
    }}

    function toggleNode(nodeId) {{
      if (state.expanded.has(nodeId)) state.expanded.delete(nodeId);
      else state.expanded.add(nodeId);
      rerenderTree();
    }}

    function findNodeById(node, nodeId) {{
      if (node.id === nodeId) return node;
      for (const c of node.children) {{
        const found = findNodeById(c, nodeId);
        if (found) return found;
      }}
      return null;
    }}

    function selectNode(node) {{
      state.selectedNodeId = node.id;
      renderDetails(node);
      for (const [id, row] of state.rowById.entries()) {{
        row.classList.toggle("selected", id === node.id);
      }}
    }}

    function formatList(values) {{
      if (!values || values.length === 0) return "[]";
      return "[" + values.join(", ") + "]";
    }}

    function renderThumbnails(cameraIndices) {{
      if (!cameraIndices.length) {{
        return `<div class="muted">No cameras in this node.</div>`;
      }}

      const clipped = cameraIndices.slice(0, MAX_THUMBS_PER_NODE);
      const cards = clipped.map((idx) => {{
        const key = String(idx);
        const entry = THUMB_DATA[key];
        if (!entry) {{
          return `<div class="thumb"><div class="thumb-caption">cam ${{idx}} (no loader data)</div></div>`;
        }}
        const caption = `cam ${{idx}}<br/>${{entry.filename || ""}}`;
        if (entry.thumbnail_data_uri) {{
          return `
            <div class="thumb">
              <img src="${{entry.thumbnail_data_uri}}" alt="cam_${{idx}}" loading="lazy" />
              <div class="thumb-caption">${{caption}}</div>
            </div>
          `;
        }}
        return `
          <div class="thumb">
            <div class="thumb-caption">${{caption}}<br/><span class="muted">thumbnail unavailable</span></div>
          </div>
        `;
      }});

      const clippedNote = cameraIndices.length > MAX_THUMBS_PER_NODE
        ? `
          <div class="muted">
            Showing first ${{MAX_THUMBS_PER_NODE}} / ${{cameraIndices.length}} thumbnails.
            Increase --max_thumbnails_per_node to show more.
          </div>
        `
        : "";

      return `${{clippedNote}}<div class="thumb-grid">${{cards.join("")}}</div>`;
    }}

    function escapeHtml(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function simToNorm(sim) {{
      if (sim === null || sim === undefined || Number.isNaN(sim)) return 0.0;
      return Math.max(0.0, Math.min(1.0, Number(sim)));
    }}

    function graphDataForNode(node, cameraIndices, edgeScope) {{
      const selectedSet = new Set(cameraIndices.map((v) => Number(v)));
      const edgeSource = edgeScope === "local" ? node.local_edges : node.subtree_edges;
      const filteredEdges = edgeSource.filter((e) => selectedSet.has(e[0]) && selectedSet.has(e[1]));
      const degree = new Map();
      for (const idx of selectedSet) degree.set(idx, 0);
      for (const [i, j] of filteredEdges) {{
        degree.set(i, (degree.get(i) || 0) + 1);
        degree.set(j, (degree.get(j) || 0) + 1);
      }}

      const sortedNodes = Array.from(selectedSet.values()).sort((a, b) => (degree.get(b) || 0) - (degree.get(a) || 0));
      const clippedNodes = sortedNodes.slice(0, MAX_GRAPH_NODES);
      const clippedSet = new Set(clippedNodes);
      const clippedEdges = filteredEdges.filter((e) => clippedSet.has(e[0]) && clippedSet.has(e[1]));
      const finiteSims = clippedEdges
        .map((e) => e[2])
        .filter((v) => v !== null && v !== undefined && !Number.isNaN(v))
        .map((v) => Number(v));
      const simMin = finiteSims.length ? Math.min(...finiteSims) : 0.0;
      const simMax = finiteSims.length ? Math.max(...finiteSims) : 1.0;
      const simRange = Math.max(1e-8, simMax - simMin);

      const elements = [];
      for (const idx of clippedNodes) {{
        const key = String(idx);
        const entry = THUMB_DATA[key] || {{}};
        elements.push({{
          data: {{
            id: `n_${{idx}}`,
            camId: idx,
            label: String(idx),
            img: entry.thumbnail_data_uri || "",
          }},
        }});
      }}
      clippedEdges.forEach(([i, j, sim], k) => {{
        const simAbsNorm = simToNorm(sim);
        const simDisplayNorm = (sim === null || sim === undefined || Number.isNaN(sim))
          ? 0.0
          : Math.max(0.0, Math.min(1.0, (Number(sim) - simMin) / simRange));
        elements.push({{
          data: {{
            id: `e_${{k}}_${{i}}_${{j}}`,
            source: `n_${{i}}`,
            target: `n_${{j}}`,
            sourceCam: i,
            targetCam: j,
            sim: sim === null || sim === undefined || Number.isNaN(sim) ? null : Number(sim),
            simLabel: sim === null || sim === undefined || Number.isNaN(sim) ? "n/a" : Number(sim).toFixed(4),
            simNorm: simAbsNorm,
            simDisplayNorm: simDisplayNorm,
          }},
        }});
      }});

      return {{
        elements,
        numNodes: clippedNodes.length,
        numEdges: clippedEdges.length,
        totalNodes: sortedNodes.length,
        simMin: simMin,
        simMax: simMax,
      }};
    }}

    function renderSimilarityGraph(node, cameraIndices, edgeScope) {{
      if (!cameraIndices.length) {{
        state.pendingGraphData = null;
        return `<div class="muted">No cameras to graph in this node.</div>`;
      }}
      const data = graphDataForNode(node, cameraIndices, edgeScope);
      state.pendingGraphData = data;
      const clipNote = data.totalNodes > MAX_GRAPH_NODES
        ? `<div class="muted">Graph clipped to top ${{MAX_GRAPH_NODES}} nodes by within-node degree (from ${{data.totalNodes}} nodes).</div>`
        : "";
      return `
        ${{clipNote}}
        <div class="graph-shell">
          <div id="cyGraph" class="graph-cy"></div>
          <div id="cyGraphTooltip" class="graph-tooltip"></div>
        </div>
        <div class="legend">
          <span>Edges: ${{data.numEdges}}</span>
          <span>Nodes: ${{data.numNodes}}</span>
          <span>sim range: ${{data.simMin.toFixed(3)}} to ${{data.simMax.toFixed(3)}}</span>
          <span>Similarity color/width</span>
          <span class="legend-bar"></span>
          <span>low → high</span>
        </div>
      `;
    }}

    function renderDetails(node) {{
      const details = document.getElementById("details");
      const selectedCameraIndices = state.thumbnailScope === "local" ? node.local_keys : node.subtree_keys;
      const selectedScopeLabel = state.thumbnailScope === "local" ? "local cameras" : "subtree cameras";
      const edgeScopeLabel = state.thumbnailScope === "local" ? "local edges" : "subtree edges";
      const graphHtml = state.showSimilarityGraph
        ? `
          <div class="card">
            <div><strong>Similarity Graph (${{selectedScopeLabel}}, ${{edgeScopeLabel}})</strong></div>
            <div class="muted">Drag nodes to adjust layout. Mouse wheel to zoom, drag empty space to pan.</div>
            ${{renderSimilarityGraph(node, selectedCameraIndices, state.thumbnailScope)}}
          </div>
        `
        : `
          <div class="card">
            <div><strong>Similarity Graph</strong></div>
            <div class="muted">Hidden. Use "Show graph" to display.</div>
          </div>
        `;
      details.innerHTML = `
        <div class="card">
          <div class="meta-grid">
            <div class="meta-item"><div class="meta-key">Node Path</div><div>${{node.path}}</div></div>
            <div class="meta-item"><div class="meta-key">Depth</div><div>${{node.depth}}</div></div>
            <div class="meta-item"><div class="meta-key">Children</div><div>${{node.num_children}}</div></div>
            <div class="meta-item"><div class="meta-key">Leaf</div><div>${{node.is_leaf}}</div></div>
            <div class="meta-item"><div class="meta-key">Local Edges</div><div>${{node.local_edge_count}}</div></div>
            <div class="meta-item"><div class="meta-key">Subtree Edges</div><div>${{node.subtree_edge_count}}</div></div>
            <div class="meta-item"><div class="meta-key">Local Cameras</div><div>${{node.local_keys.length}}</div></div>
            <div class="meta-item">
              <div class="meta-key">Subtree Cameras</div>
              <div>${{node.subtree_keys.length}}</div>
            </div>
          </div>
        </div>
        <div class="card">
          <div><strong>Local camera indices</strong></div>
          <div class="codebox">${{formatList(node.local_keys)}}</div>
        </div>
        <div class="card">
          <div><strong>Subtree camera indices</strong></div>
          <div class="codebox">${{formatList(node.subtree_keys)}}</div>
        </div>
        <div class="card">
          <div><strong>Thumbnails (${{selectedScopeLabel}})</strong></div>
          ${{renderThumbnails(selectedCameraIndices)}}
        </div>
        ${{graphHtml}}
      `;
      initializeSimilarityGraphCytoscape();
    }}

    function rerenderTree() {{
      state.rowById.clear();
      state.childrenUlById.clear();
      const rootEl = document.getElementById("treeRoot");
      rootEl.innerHTML = "";
      rootEl.appendChild(createNodeElement(TREE_DATA));

      // Apply camera-index filter by dimming non-matching nodes.
      for (const [id, row] of state.rowById.entries()) {{
        const node = findNodeById(TREE_DATA, id);
        if (!node) continue;
        const visible = subtreeContainsCamera(node, state.searchCameraIdx);
        row.style.opacity = visible ? "1.0" : "0.25";
      }}

      // Restore selected row style if possible.
      if (state.selectedNodeId) {{
        const selectedNode = findNodeById(TREE_DATA, state.selectedNodeId);
        if (selectedNode) selectNode(selectedNode);
      }}
    }}

    function initializeSimilarityGraphCytoscape() {{
      const container = document.getElementById("cyGraph");
      if (!container || !state.pendingGraphData) {{
        if (state.cy) {{
          state.cy.destroy();
          state.cy = null;
        }}
        return;
      }}
      if (typeof cytoscape === "undefined") {{
        container.innerHTML = '<div class="muted" style="padding:10px;">Cytoscape failed to load.</div>';
        return;
      }}

      if (state.cy) {{
        state.cy.destroy();
        state.cy = null;
      }}

      const tooltip = document.getElementById("cyGraphTooltip");
      const cy = cytoscape({{
        container: container,
        elements: state.pendingGraphData.elements,
        style: [
          {{
            selector: "node",
            style: {{
              "width": 56,
              "height": 56,
              "shape": "round-rectangle",
              "background-color": "#111827",
              "background-image": "data(img)",
              "background-fit": "cover",
              "background-clip": "node",
              "border-color": "#374151",
              "border-width": 1.2,
              "label": "data(label)",
              "font-size": 9,
              "color": "#e5e7eb",
              "text-halign": "center",
              "text-valign": "bottom",
              "text-margin-y": 10,
              "text-background-color": "rgba(2, 6, 23, 0.75)",
              "text-background-opacity": 1,
              "text-background-padding": 2,
              "overlay-opacity": 0,
            }},
          }},
          {{
            selector: "edge",
            style: {{
              "width": "mapData(simDisplayNorm, 0, 1, 1, 4)",
              "line-color": "mapData(simDisplayNorm, 0, 1, #64748b, #f97316)",
              "opacity": 0.85,
              "curve-style": "bezier",
            }},
          }},
          {{
            selector: "node:selected",
            style: {{
              "border-color": "#60a5fa",
              "border-width": 2.5,
            }},
          }},
        ],
        layout: {{
          name: "cose",
          animate: false,
          fit: true,
          padding: 20,
          nodeRepulsion: 180000,
          idealEdgeLength: (edge) => 70 + (1 - edge.data("simNorm")) * 90,
          edgeElasticity: (edge) => 90 + edge.data("simNorm") * 120,
          gravity: 0.4,
          numIter: 1000,
        }},
        minZoom: 0.2,
        maxZoom: 4.0,
      }});
      state.cy = cy;

      function placeTooltip(evt, text) {{
        if (!tooltip) return;
        const rect = container.getBoundingClientRect();
        const rendered = evt.renderedPosition || {{x: 0, y: 0}};
        tooltip.textContent = text;
        tooltip.style.display = "block";
        tooltip.style.left = `${{Math.max(8, Math.min(rect.width - 140, rendered.x + 10))}}px`;
        tooltip.style.top = `${{Math.max(8, Math.min(rect.height - 28, rendered.y + 10))}}px`;
      }}

      cy.on("mouseover", "edge", (evt) => {{
        const e = evt.target;
        placeTooltip(evt, `sim(${{e.data("sourceCam")}},${{e.data("targetCam")}}) = ${{e.data("simLabel")}}`);
      }});
      cy.on("mousemove", "edge", (evt) => {{
        const e = evt.target;
        placeTooltip(evt, `sim(${{e.data("sourceCam")}},${{e.data("targetCam")}}) = ${{e.data("simLabel")}}`);
      }});
      cy.on("mouseout", "edge", () => {{
        if (!tooltip) return;
        tooltip.style.display = "none";
      }});
    }}

    document.getElementById("expandAllBtn").addEventListener("click", () => {{
      for (const id of allNodeIds(TREE_DATA)) {{
        state.expanded.add(id);
      }}
      rerenderTree();
    }});

    document.getElementById("collapseAllBtn").addEventListener("click", () => {{
      state.expanded.clear();
      state.expanded.add("root");
      rerenderTree();
    }});

    document.getElementById("searchInput").addEventListener("input", (evt) => {{
      const text = evt.target.value.trim();
      if (text === "") {{
        state.searchCameraIdx = null;
      }} else {{
        const parsed = Number.parseInt(text, 10);
        state.searchCameraIdx = Number.isNaN(parsed) ? null : parsed;
      }}
      rerenderTree();
    }});

    document.getElementById("thumbScopeSelect").addEventListener("change", (evt) => {{
      state.thumbnailScope = evt.target.value === "local" ? "local" : "subtree";
      if (state.selectedNodeId) {{
        const selectedNode = findNodeById(TREE_DATA, state.selectedNodeId);
        if (selectedNode) renderDetails(selectedNode);
      }}
    }});

    function updateGraphToggleButton() {{
      const btn = document.getElementById("toggleGraphBtn");
      if (!btn) return;
      btn.textContent = state.showSimilarityGraph ? "Hide graph" : "Show graph";
    }}

    document.getElementById("toggleGraphBtn").addEventListener("click", () => {{
      state.showSimilarityGraph = !state.showSimilarityGraph;
      updateGraphToggleButton();
      if (state.selectedNodeId) {{
        const selectedNode = findNodeById(TREE_DATA, state.selectedNodeId);
        if (selectedNode) renderDetails(selectedNode);
      }}
    }});

    updateGraphToggleButton();
    rerenderTree();
    selectNode(TREE_DATA);
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster_tree_pkl",
        type=str,
        required=True,
        help="Path to serialized ClusterTree pickle.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default=None,
        help="Output HTML path. Default: <cluster_tree_pkl parent>/cluster_tree_viz.html",
    )
    parser.add_argument(
        "--max_thumbnails_per_node",
        type=int,
        default=240,
        help="UI display cap on thumbnails shown per selected node.",
    )
    parser.add_argument(
        "--thumbnail_size",
        type=int,
        default=256,
        help="Thumbnail max side length in pixels (loader mode only).",
    )
    parser.add_argument(
        "--similarity_matrix_path",
        type=str,
        default=None,
        help=(
            "Optional path to similarity_matrix.txt (CSV). "
            "If omitted, tries <cluster_tree_pkl parent>/plots/similarity_matrix.txt."
        ),
    )
    parser.add_argument(
        "--cytoscape_js_path",
        type=str,
        default=str((Path(__file__).resolve().parent / "assets" / "cytoscape.min.js")),
        help="Path to local cytoscape.min.js bundle for offline interactive graph rendering.",
    )

    # Optional loader arguments for thumbnails.
    parser.add_argument(
        "--loader_config",
        type=str,
        default=None,
        help="Loader config in gtsfm/configs/loader (e.g. olsson, colmap). If omitted, no thumbnails are loaded.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset root for loader instantiation (required with --loader_config).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional image directory override for loader.",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=None,
        help="Optional loader max_resolution override.",
    )
    parser.add_argument(
        "--colmap_files_subdir",
        type=str,
        default=None,
        help="Optional ColmapLoader arg: subdirectory under dataset_dir containing COLMAP txt files (e.g. sfm).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    cluster_tree_pkl = Path(args.cluster_tree_pkl).resolve()
    if not cluster_tree_pkl.exists():
        raise FileNotFoundError(f"cluster_tree_pkl does not exist: {cluster_tree_pkl}")

    output_html = (
        Path(args.output_html).resolve()
        if args.output_html is not None
        else cluster_tree_pkl.parent / "cluster_tree_viz.html"
    )

    with cluster_tree_pkl.open("rb") as f:
        cluster_tree = pickle.load(f)
    if not isinstance(cluster_tree, ClusterTree):
        raise TypeError(
            f"Expected pickled object to be ClusterTree, got: {type(cluster_tree)}. "
            "If this is from an older run, verify cluster_tree.pkl is from graph partitioning."
        )

    similarity_matrix_path = (
        Path(args.similarity_matrix_path).resolve()
        if args.similarity_matrix_path is not None
        else cluster_tree_pkl.parent / "plots" / "similarity_matrix.txt"
    )
    similarity_matrix = _load_similarity_matrix(similarity_matrix_path)
    cytoscape_source = _load_cytoscape_source(Path(args.cytoscape_js_path).resolve())

    tree_data = _tree_to_dict(
        node=cluster_tree,
        node_id="root",
        path=(),
        depth=0,
        similarity_matrix=similarity_matrix,
    )
    all_camera_indices = sorted(_collect_all_camera_indices(tree_data))
    logger.info(
        "Loaded cluster tree with %d total cameras in subtree union.",
        len(all_camera_indices),
    )

    thumbnail_payload: Optional[Dict[str, Dict[str, str]]] = None
    if args.loader_config is not None:
        if args.dataset_dir is None:
            raise ValueError("--dataset_dir is required when --loader_config is provided.")
        logger.info("Building loader config=%s for thumbnails.", args.loader_config)
        hydra_available = importlib.util.find_spec("hydra") is not None
        if hydra_available:
            loader = _build_loader(
                loader_config=args.loader_config,
                dataset_dir=args.dataset_dir,
                images_dir=args.images_dir,
                max_resolution=args.max_resolution,
                colmap_files_subdir=args.colmap_files_subdir,
            )
            logger.info("Rendering thumbnails for %d camera indices.", len(all_camera_indices))
            thumbnail_payload = _build_thumbnail_payload(
                loader=loader,
                camera_indices=all_camera_indices,
                thumb_size_px=args.thumbnail_size,
            )
        elif args.loader_config == "colmap":
            dataset_dir = Path(args.dataset_dir).resolve()
            images_dir = Path(args.images_dir).resolve() if args.images_dir is not None else dataset_dir / "images"
            colmap_dir = dataset_dir / args.colmap_files_subdir if args.colmap_files_subdir is not None else dataset_dir
            logger.warning(
                "Hydra is unavailable; using Colmap fallback thumbnail loading from %s and %s.",
                colmap_dir,
                images_dir,
            )
            thumbnail_payload = _build_colmap_thumbnail_payload_without_loader(
                camera_indices=all_camera_indices,
                images_dir=images_dir,
                colmap_files_dir=colmap_dir,
                thumb_size_px=args.thumbnail_size,
            )
        elif args.loader_config == "olsson":
            dataset_dir = Path(args.dataset_dir).resolve()
            images_dir = Path(args.images_dir).resolve() if args.images_dir is not None else None
            logger.warning(
                "Hydra is unavailable; using Olsson fallback thumbnail loading from dataset=%s images_dir=%s.",
                dataset_dir,
                images_dir if images_dir is not None else dataset_dir / "images",
            )
            thumbnail_payload = _build_olsson_thumbnail_payload_without_loader(
                camera_indices=all_camera_indices,
                dataset_dir=dataset_dir,
                images_dir=images_dir,
                thumb_size_px=args.thumbnail_size,
            )
        else:
            raise ImportError(
                "Hydra is not installed, and fallback is only available for loader_config in {colmap, olsson}."
            )
    else:
        logger.info("No loader specified; HTML will include tree + index metadata only.")

    _render_html(
        tree_data=tree_data,
        output_html=output_html,
        thumbnail_payload=thumbnail_payload,
        max_thumbnails_per_node=args.max_thumbnails_per_node,
        cytoscape_source=cytoscape_source,
    )
    logger.info("Saved cluster tree visualization to %s", output_html)


if __name__ == "__main__":
    main()
