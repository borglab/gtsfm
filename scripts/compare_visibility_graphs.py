"""Compare MegaLoc visibility graph vs GT COLMAP/GLOMAP covisibility graph.

Computes precision, recall, F1, graph stats, fragmentation metrics, and PR curves.

Usage:
    # With MegaLoc results + PR curves:
    python scripts/compare_visibility_graphs.py \
        --dataset_name Gendarmenmarkt \
        --colmap_dir benchmarks/Gendarmenmarkt/sparse_glomap/0 \
        --images_dir benchmarks/Gendarmenmarkt/images \
        --megaloc_pairs_file results/gendermarket_results_2/plots/similarity_named_pairs.txt \
        --similarity_matrix results/gendermarket_results_2/plots/similarity_matrix.txt \
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def load_similarity_matrix(matrix_path: str, num_images: int) -> dict[tuple[int, int], float]:
    """Load full N×N similarity matrix and return upper-triangle scores.

    Returns:
        Dict mapping (i, j) with i < j to similarity score (only entries > 0).
    """
    mat = np.loadtxt(matrix_path, delimiter=",")
    assert mat.shape[0] == mat.shape[1], f"Expected square matrix, got {mat.shape}"
    if mat.shape[0] != num_images:
        print(f"  WARNING: similarity matrix has {mat.shape[0]} images, expected {num_images}")
    scores: dict[tuple[int, int], float] = {}
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] > 0 and np.isfinite(mat[i, j]):
                scores[(i, j)] = float(mat[i, j])
    return scores


def compute_pr_curve_sweep_score(
    all_scores: dict[tuple[int, int], float],
    gt_set: set[tuple[int, int]],
    num_thresholds: int = 50,
) -> list[dict]:
    """Sweep MegaLoc score threshold and compute precision/recall at each point."""
    # Sort edges by score descending for efficient sweep.
    sorted_edges = sorted(all_scores.items(), key=lambda x: -x[1])
    thresholds = np.linspace(0.0, 1.0, num_thresholds + 1).tolist()

    curve = []
    for t in thresholds:
        predicted = {edge for edge, score in sorted_edges if score >= t}
        tp = len(predicted & gt_set)
        precision = tp / len(predicted) if predicted else 1.0
        recall = tp / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        curve.append({
            "threshold": round(t, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "num_predicted": len(predicted),
        })
    return curve


def compute_pr_curve_sweep_gt(
    megaloc_set: set[tuple[int, int]],
    pair_counts: dict[tuple[int, int], int],
    max_threshold: int = 100,
) -> list[dict]:
    """Sweep GT covisibility threshold and compute precision/recall at each point."""
    curve = []
    for min_pts in range(1, max_threshold + 1):
        gt_set = {edge for edge, count in pair_counts.items() if count >= min_pts}
        tp = len(megaloc_set & gt_set)
        precision = tp / len(megaloc_set) if megaloc_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        curve.append({
            "min_shared_points": min_pts,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "num_gt_edges": len(gt_set),
        })
    return curve


def plot_pr_curves(
    curve_score: list[dict] | None,
    curve_gt: list[dict] | None,
    dataset_name: str,
    output_path: str,
) -> None:
    """Plot PR curves and save to PNG."""
    num_plots = (1 if curve_score else 0) + (1 if curve_gt else 0)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    if curve_score:
        ax = axes[plot_idx]
        recalls = [p["recall"] for p in curve_score]
        precisions = [p["precision"] for p in curve_score]
        thresholds = [p["threshold"] for p in curve_score]

        # AUC via trapezoidal rule (sort by recall ascending).
        sorted_pairs = sorted(zip(recalls, precisions))
        r_sorted = [x[0] for x in sorted_pairs]
        p_sorted = [x[1] for x in sorted_pairs]
        auc = float(np.trapezoid(p_sorted, r_sorted))

        ax.plot(recalls, precisions, "b.-", markersize=3)

        # Mark operating points at score=0.5 and best F1.
        best_f1_entry = max(curve_score, key=lambda x: x["f1"])
        ax.plot(best_f1_entry["recall"], best_f1_entry["precision"], "r*", markersize=12,
                label=f"Best F1={best_f1_entry['f1']:.3f} @ score={best_f1_entry['threshold']:.2f}")

        # Find score=0.5 point.
        for entry in curve_score:
            if abs(entry["threshold"] - 0.5) < 0.02:
                ax.plot(entry["recall"], entry["precision"], "go", markersize=8,
                        label=f"score=0.5 (P={entry['precision']:.2f}, R={entry['recall']:.2f})")
                break

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{dataset_name}: MegaLoc Score Sweep (AUC={auc:.3f})")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    if curve_gt:
        ax = axes[plot_idx]
        gt_thresholds = [p["min_shared_points"] for p in curve_gt]
        precisions = [p["precision"] for p in curve_gt]
        recalls = [p["recall"] for p in curve_gt]
        f1s = [p["f1"] for p in curve_gt]

        ax.plot(gt_thresholds, precisions, "b-", label="Precision")
        ax.plot(gt_thresholds, recalls, "r-", label="Recall")
        ax.plot(gt_thresholds, f1s, "g--", label="F1")
        ax.set_xlabel("GT min_shared_points")
        ax.set_ylabel("Score")
        ax.set_title(f"{dataset_name}: GT Threshold Sweep")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nPR curve plot saved to {output_path}")


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
    parser.add_argument("--similarity_matrix", default=None, help="Path to similarity_matrix.txt (full N×N CSV) for PR curve sweep")
    parser.add_argument("--min_shared_points", type=int, default=30, help="GT covisibility threshold")
    parser.add_argument("--output_json", default=None, help="Path to write results JSON")
    parser.add_argument("--output_plot", default=None, help="Path to save PR curve PNG")
    parser.add_argument("--num_score_thresholds", type=int, default=50, help="Number of MegaLoc score thresholds to sweep")
    parser.add_argument("--max_gt_threshold", type=int, default=100, help="Max GT min_shared_points for sweep")
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
    megaloc_set = None
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

    # --- PR Curves ---
    curve_score = None
    curve_gt = None
    megaloc_set_for_gt_sweep = None

    if args.similarity_matrix:
        print("\n" + "=" * 60)
        print("PR CURVE: SWEEP MEGALOC SCORE THRESHOLD")
        print("=" * 60)
        all_scores = load_similarity_matrix(args.similarity_matrix, len(image_fnames))
        print(f"  Loaded {len(all_scores)} pairwise scores from similarity matrix")

        curve_score = compute_pr_curve_sweep_score(all_scores, gt_set, args.num_score_thresholds)
        results["pr_curve_sweep_score"] = curve_score

        # AUC
        sorted_pairs = sorted([(p["recall"], p["precision"]) for p in curve_score])
        auc = float(np.trapezoid([x[1] for x in sorted_pairs], [x[0] for x in sorted_pairs]))
        results["pr_curve_sweep_score_auc"] = round(auc, 4)

        # Best F1
        best = max(curve_score, key=lambda x: x["f1"])
        results["best_f1_score_threshold"] = best["threshold"]
        results["best_f1"] = best["f1"]
        results["best_f1_precision"] = best["precision"]
        results["best_f1_recall"] = best["recall"]
        results["best_f1_num_predicted"] = best["num_predicted"]
        print(f"  AUC: {auc:.4f}")
        print(f"  Best F1: {best['f1']:.4f} at score threshold {best['threshold']:.2f}")
        print(f"    Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}")
        print(f"    Predicted edges: {best['num_predicted']}, GT edges: {len(gt_set)}")

        # Stats at score=0.5 (our current operating point).
        pt05 = min(curve_score, key=lambda x: abs(x["threshold"] - 0.5))
        print(f"  At score=0.5: P={pt05['precision']:.4f}, R={pt05['recall']:.4f}, "
              f"F1={pt05['f1']:.4f}, edges={pt05['num_predicted']}")

        # Derive megaloc_set at 0.5 for GT sweep if no pairs file was provided.
        if not args.megaloc_pairs_file:
            megaloc_set_for_gt_sweep = {edge for edge, score in all_scores.items() if score >= 0.5}

    # Use megaloc_set from pairs file, or derive from similarity matrix at score=0.5.
    gt_sweep_set = megaloc_set if megaloc_set is not None else megaloc_set_for_gt_sweep
    if gt_sweep_set is not None:
        print("\n" + "=" * 60)
        print("PR CURVE: SWEEP GT THRESHOLD")
        print("=" * 60)
        sweep_set = gt_sweep_set
        curve_gt = compute_pr_curve_sweep_gt(sweep_set, pair_counts, args.max_gt_threshold)
        results["pr_curve_sweep_gt"] = curve_gt
        best_gt = max(curve_gt, key=lambda x: x["f1"])
        print(f"  Best F1: {best_gt['f1']:.4f} at min_shared_points={best_gt['min_shared_points']}")
        print(f"    Precision: {best_gt['precision']:.4f}, Recall: {best_gt['recall']:.4f}")
        print(f"    GT edges at that threshold: {best_gt['num_gt_edges']}")

    # Plot
    if curve_score or curve_gt:
        output_plot = args.output_plot
        if not output_plot and args.output_json:
            output_plot = args.output_json.replace(".json", "_pr.png")
        if output_plot:
            plot_pr_curves(curve_score, curve_gt, args.dataset_name, output_plot)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"SUMMARY: {args.dataset_name}")
    print("=" * 60)
    print(f"  Images on disk: {len(image_fnames)}")
    print(f"  GT registered: {num_registered} (with 3D obs: {num_with_obs})")
    print(f"  GT graph (min_shared={args.min_shared_points}): {gt_stats['num_nodes']} nodes, "
          f"{gt_stats['num_edges']} edges, {gt_stats['num_components']} components")
    gt_density = gt_stats['num_edges'] / (gt_stats['num_nodes'] * (gt_stats['num_nodes'] - 1) / 2) * 100 if gt_stats['num_nodes'] > 1 else 0
    print(f"  GT graph density: {gt_density:.1f}%")
    if curve_score:
        print(f"  Score sweep AUC: {results['pr_curve_sweep_score_auc']}")
        print(f"  Best F1: {best['f1']:.4f} @ score={best['threshold']:.2f} "
              f"(P={best['precision']:.4f}, R={best['recall']:.4f}, {best['num_predicted']} edges)")
        print(f"  At score=0.5: P={pt05['precision']:.4f}, R={pt05['recall']:.4f}, "
              f"F1={pt05['f1']:.4f}, {pt05['num_predicted']} edges")
    if curve_gt:
        print(f"  GT sweep best F1: {best_gt['f1']:.4f} @ min_shared={best_gt['min_shared_points']} "
              f"(P={best_gt['precision']:.4f}, R={best_gt['recall']:.4f})")

    # --- Write JSON ---
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
