"""Compare MegaLoc visibility graph vs GT COLMAP/GLOMAP covisibility graph.

Computes precision, recall, F1, graph stats, and fragmentation metrics.

Usage:
    # With MegaLoc results:
    python scripts/compare_visibility_graphs.py \
        --dataset_name Gendarmenmarkt \
        --colmap_dir benchmarks/Gendarmenmarkt/sparse_glomap/0 \
        --images_dir benchmarks/Gendarmenmarkt/images \
        --megaloc_pairs_file results/gendermarket_results_2/plots/similarity_named_pairs.txt \
        --min_shared_points 30 \
        --output_json results/analysis/Gendarmenmarkt.json

    # GT-only stats (no MegaLoc comparison):
    python scripts/compare_visibility_graphs.py \
        --dataset_name Gendarmenmarkt \
        --colmap_dir benchmarks/Gendarmenmarkt/sparse_glomap/0 \
        --images_dir benchmarks/Gendarmenmarkt/images \
        --min_shared_points 30 \
        --output_json results/analysis/Gendarmenmarkt.json
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

import thirdparty.colmap.scripts.python.read_write_model as colmap_io


def connected_components(adj: dict[int, set[int]]) -> list[set[int]]:
    """Return connected components of an undirected graph given as adjacency dict."""
    visited: set[int] = set()
    components: list[set[int]] = []
    for n in adj:
        if n not in visited:
            comp: set[int] = set()
            queue = [n]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.add(curr)
                queue.extend(adj[curr] - visited)
            components.append(comp)
    return components


def graph_stats(pairs: list[tuple[int, int]], label: str) -> dict:
    """Compute and print graph statistics."""
    if not pairs:
        print(f"\n[{label}] Empty graph!")
        return {"num_nodes": 0, "num_edges": 0, "num_components": 0}

    adj: defaultdict[int, set[int]] = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)

    nodes = sorted(adj.keys())
    degrees = [len(adj[n]) for n in nodes]
    components = connected_components(adj)
    comp_sizes = sorted([len(c) for c in components], reverse=True)

    stats = {
        "num_nodes": len(nodes),
        "num_edges": len(pairs),
        "min_degree": int(min(degrees)),
        "max_degree": int(max(degrees)),
        "mean_degree": round(float(np.mean(degrees)), 1),
        "median_degree": int(np.median(degrees)),
        "num_components": len(components),
        "largest_component": comp_sizes[0],
        "component_sizes": comp_sizes[:10],
    }

    print(f"\n[{label}]")
    print(f"  Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    print(f"  Degree: min={stats['min_degree']}, max={stats['max_degree']}, "
          f"mean={stats['mean_degree']}, median={stats['median_degree']}")
    print(f"  Components: {stats['num_components']} (largest: {stats['largest_component']})")
    if len(comp_sizes) > 1:
        print(f"  Top component sizes: {comp_sizes[:10]}")

    return stats


def load_megaloc_pairs(
    pairs_file: str, fname_to_idx: dict[str, int]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float]]:
    """Load MegaLoc pairs from similarity_named_pairs.txt.

    Returns:
        pairs: deduplicated list of (i, j) with i < j
        scores: dict mapping edge -> similarity score
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    scores: dict[tuple[int, int], float] = {}

    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            score = float(parts[0])
            fname_a = Path(parts[1]).name
            fname_b = Path(parts[2]).name
            idx_a = fname_to_idx.get(fname_a)
            idx_b = fname_to_idx.get(fname_b)
            if idx_a is not None and idx_b is not None:
                edge = (min(idx_a, idx_b), max(idx_a, idx_b))
                if edge not in seen:
                    seen.add(edge)
                    pairs.append(edge)
                scores[edge] = max(scores.get(edge, 0), score)

    return pairs, scores


def build_gt_pair_counts(
    colmap_dir: str, image_fnames: list[str]
) -> tuple[dict[tuple[int, int], int], int, int, int]:
    """Build pairwise shared-point counts from COLMAP/GLOMAP points3D.

    Returns:
        pair_counts: dict mapping (i, j) -> number of shared 3D points
        num_registered: number of COLMAP images
        num_with_obs: number of images appearing in at least one 3D point track
        num_points3d: number of 3D points
    """
    colmap_path = Path(colmap_dir)
    if (colmap_path / "images.txt").exists():
        ext = ".txt"
    elif (colmap_path / "images.bin").exists():
        ext = ".bin"
    else:
        raise FileNotFoundError(f"No COLMAP images file found in {colmap_path}")

    _, images, points3d = colmap_io.read_model(path=str(colmap_path), ext=ext)

    fname_to_idx = {fname: idx for idx, fname in enumerate(image_fnames)}
    cid_to_lid: dict[int, int] = {}
    for img in images.values():
        basename = Path(img.name).name
        if basename in fname_to_idx:
            cid_to_lid[img.id] = fname_to_idx[basename]

    num_with_obs = sum(
        1 for img in images.values() if any(pid > 0 for pid in img.point3D_ids)
    )

    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for pt in points3d.values():
        lids = set()
        for cid in pt.image_ids:
            lid = cid_to_lid.get(int(cid))
            if lid is not None:
                lids.add(lid)
        for a, b in combinations(sorted(lids), 2):
            pair_counts[(a, b)] += 1

    return dict(pair_counts), len(images), num_with_obs, len(points3d)


def main():
    parser = argparse.ArgumentParser(
        description="Compare MegaLoc vs GT COLMAP/GLOMAP covisibility graphs"
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset label (e.g. Gendarmenmarkt)")
    parser.add_argument("--colmap_dir", required=True, help="Path to COLMAP/GLOMAP sparse model directory")
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--megaloc_pairs_file", default=None, help="Path to similarity_named_pairs.txt")
    parser.add_argument("--min_shared_points", type=int, default=30, help="GT covisibility threshold")
    parser.add_argument("--output_json", default=None, help="Path to write results JSON")
    args = parser.parse_args()

    # Get image filenames (sorted, basename only).
    images_dir = Path(args.images_dir)
    image_fnames = sorted([
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    fname_to_idx = {fname: idx for idx, fname in enumerate(image_fnames)}
    print(f"Dataset: {args.dataset_name}")
    print(f"Images on disk: {len(image_fnames)}")

    # --- Build GT covisibility ---
    print("\nBuilding GT covisibility from COLMAP/GLOMAP points3D...")
    pair_counts, num_registered, num_with_obs, num_points3d = build_gt_pair_counts(
        args.colmap_dir, image_fnames
    )

    gt_pairs = [
        edge for edge, count in pair_counts.items()
        if count >= args.min_shared_points
    ]
    gt_pairs.sort()
    gt_set = set(gt_pairs)

    print(f"  Registered images: {num_registered} (with 3D obs: {num_with_obs})")
    print(f"  3D points: {num_points3d}")
    print(f"  Total covisible pairs (any threshold): {len(pair_counts)}")

    gt_stats = graph_stats(gt_pairs, f"GT Covisibility (min_shared={args.min_shared_points})")

    # Compute GLOMAP-style tau
    counts_above_5 = [c for c in pair_counts.values() if c >= 5]
    tau = float(np.median(counts_above_5)) if counts_above_5 else 0
    print(f"  GLOMAP-style tau (median of pairs>=5): {tau}, 0.75*tau: {0.75*tau:.1f}")

    results = {
        "dataset_name": args.dataset_name,
        "num_images_on_disk": len(image_fnames),
        "num_registered": num_registered,
        "num_with_3d_obs": num_with_obs,
        "num_3d_points": num_points3d,
        "min_shared_points": args.min_shared_points,
        "glomap_tau": tau,
        "gt": gt_stats,
    }

    # --- MegaLoc comparison ---
    if args.megaloc_pairs_file:
        print("\n" + "=" * 60)
        print("MEGALOC COMPARISON")
        print("=" * 60)

        megaloc_pairs, megaloc_scores = load_megaloc_pairs(args.megaloc_pairs_file, fname_to_idx)
        megaloc_set = set(megaloc_pairs)

        megaloc_stats = graph_stats(megaloc_pairs, "MegaLoc")
        results["megaloc"] = megaloc_stats

        # Precision / Recall / F1
        true_positives = megaloc_set & gt_set
        false_positives = megaloc_set - gt_set
        false_negatives = gt_set - megaloc_set

        precision = len(true_positives) / len(megaloc_set) if megaloc_set else 0.0
        recall = len(true_positives) / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n--- Precision / Recall ---")
        print(f"  True positives (shared edges): {len(true_positives)}")
        print(f"  False positives (MegaLoc only): {len(false_positives)}")
        print(f"  False negatives (GT only):      {len(false_negatives)}")
        print(f"  Precision: {precision:.3f} ({100*precision:.1f}%)")
        print(f"  Recall:    {recall:.3f} ({100*recall:.1f}%)")
        print(f"  F1:        {f1:.3f}")

        results["precision"] = round(precision, 4)
        results["recall"] = round(recall, 4)
        results["f1"] = round(f1, 4)
        results["true_positives"] = len(true_positives)
        results["false_positives"] = len(false_positives)
        results["false_negatives"] = len(false_negatives)

        # Node coverage
        gt_nodes = set()
        for i, j in gt_pairs:
            gt_nodes.add(i)
            gt_nodes.add(j)
        megaloc_nodes = set()
        for i, j in megaloc_pairs:
            megaloc_nodes.add(i)
            megaloc_nodes.add(j)

        node_recall = len(megaloc_nodes & gt_nodes) / len(gt_nodes) if gt_nodes else 0.0
        print(f"\n--- Node Coverage ---")
        print(f"  GT nodes:     {len(gt_nodes)}")
        print(f"  MegaLoc nodes: {len(megaloc_nodes)}")
        print(f"  Shared nodes:  {len(megaloc_nodes & gt_nodes)}")
        print(f"  MegaLoc-only:  {len(megaloc_nodes - gt_nodes)}")
        print(f"  GT-only:       {len(gt_nodes - megaloc_nodes)}")
        print(f"  Node recall:   {node_recall:.3f} ({100*node_recall:.1f}%)")

        results["node_recall"] = round(node_recall, 4)
        results["megaloc_nodes"] = len(megaloc_nodes)
        results["gt_nodes"] = len(gt_nodes)

        # False positive breakdown: how many GT shared points do MegaLoc-only edges have?
        fp_gt_counts = [pair_counts.get(e, 0) for e in false_positives]
        buckets = {
            "0_shared": sum(1 for c in fp_gt_counts if c == 0),
            "1_to_4_shared": sum(1 for c in fp_gt_counts if 1 <= c < 5),
            "5_to_29_shared": sum(1 for c in fp_gt_counts if 5 <= c < args.min_shared_points),
            f"{args.min_shared_points}+_shared": sum(1 for c in fp_gt_counts if c >= args.min_shared_points),
        }
        print(f"\n--- False Positive Breakdown (MegaLoc-only edges) ---")
        for bucket, count in buckets.items():
            pct = 100 * count / len(false_positives) if false_positives else 0
            print(f"  {bucket}: {count} ({pct:.1f}%)")

        results["fp_breakdown"] = buckets

        # Similarity score stats for TP vs FP
        tp_scores = [megaloc_scores.get(e, 0) for e in true_positives]
        fp_scores = [megaloc_scores.get(e, 0) for e in false_positives]
        if tp_scores and fp_scores:
            print(f"\n--- Similarity Scores ---")
            print(f"  True positives:  mean={np.mean(tp_scores):.3f}, median={np.median(tp_scores):.3f}")
            print(f"  False positives: mean={np.mean(fp_scores):.3f}, median={np.median(fp_scores):.3f}")
            results["tp_score_mean"] = round(float(np.mean(tp_scores)), 4)
            results["fp_score_mean"] = round(float(np.mean(fp_scores)), 4)

    # --- Write JSON ---
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
