"""Compare saved factor graphs between two result subfolders.

Example:
    python scripts/compare_factor_graphs.py \
        --results_root /path/to/results \
        --subdir_a ba_with_gcm \
        --subdir_b ba_gcm_from_gt
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def _iter_base_dirs(results_root: Path) -> Iterable[Path]:
    yield results_root
    for metrics_dir in sorted(results_root.rglob("metrics")):
        parent = metrics_dir.parent
        if parent != results_root:
            yield parent


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_FACTOR_LINE = re.compile(r"^\s*Factor\s+(\d+):\s*(.*)$")
_KEY_TOKEN = re.compile(r"\b[A-Za-z]\d+\b")


def _parse_factors(text: str) -> list[tuple[str, tuple[str, ...], str]]:
    """Parse factors as full serialized blocks.

    We group contiguous lines by factor id, so each parsed item contains the full
    factor block (including `.z[...]` measurements and all numeric values).
    """
    factors: list[tuple[str, tuple[str, ...], str]] = []
    current_factor_id: int | None = None
    current_factor_headers: list[str] = []
    current_block_lines: list[str] = []

    def _flush_current_block() -> None:
        nonlocal current_factor_headers, current_block_lines
        if not current_block_lines:
            return
        header = next(
            (h for h in current_factor_headers if h and not h.startswith("keys =") and not h.startswith(".z[")),
            current_factor_headers[0] if current_factor_headers else "Factor",
        )
        block_text = "\n".join(current_block_lines).strip()
        keys = tuple(_KEY_TOKEN.findall(block_text))
        factors.append((header, keys, block_text))
        current_factor_headers = []
        current_block_lines = []

    for line in text.splitlines():
        match = _FACTOR_LINE.match(line)
        if match:
            factor_id = int(match.group(1))
            factor_header = match.group(2).strip()
            if current_factor_id is None:
                current_factor_id = factor_id
            elif factor_id != current_factor_id:
                _flush_current_block()
                current_factor_id = factor_id
            current_factor_headers.append(factor_header)
            current_block_lines.append(line)
            continue
        if current_factor_id is not None:
            current_block_lines.append(line)

    _flush_current_block()
    return factors


def _log_factor_summary(label: str, path: Path, factors: list[tuple[str, tuple[str, ...], str]]) -> None:
    unique_keys = {key for _, keys, _ in factors for key in keys}
    factor_type_counts = Counter(ftype for ftype, _, _ in factors)
    factor_count = len(factors)
    variable_count = len(unique_keys)
    density = (factor_count / variable_count) if variable_count else 0.0

    logger.info("%s summary: %s", label, path)
    logger.info("  - factors: %d", factor_count)
    logger.info("  - variables (unique keys): %d", variable_count)
    logger.info("  - factor-to-variable ratio: %.2f", density)
    logger.info("  - distinct factor types: %d", len(factor_type_counts))
    if factor_type_counts:
        logger.info("  - most common factor types (top 5):")
        for factor_type, count in factor_type_counts.most_common(5):
            percent = (100.0 * count / factor_count) if factor_count else 0.0
            logger.info("      * %s: %d (%.1f%%)", factor_type, count, percent)
    else:
        logger.info("  - most common factor types (top 5): <none>")


def _compare_file(a_path: Path, b_path: Path) -> bool:
    a_data = _read_bytes(a_path)
    b_data = _read_bytes(b_path)

    a_text = a_data.decode("utf-8", errors="ignore")
    b_text = b_data.decode("utf-8", errors="ignore")
    a_factors = _parse_factors(a_text)
    b_factors = _parse_factors(b_text)
    _log_factor_summary("A", a_path, a_factors)
    _log_factor_summary("B", b_path, b_factors)

    if a_data == b_data:
        logger.info("Match: %s vs %s (byte-identical)", a_path, b_path)
        return True

    if not a_factors or not b_factors:
        logger.info(
            "Mismatch: %s vs %s (unable to parse factors; parsed_factors=%d/%d)",
            a_path,
            b_path,
            len(a_factors),
            len(b_factors),
        )
        return False

    if len(a_factors) != len(b_factors):
        logger.info(
            "Mismatch: %s vs %s (factor_count %d vs %d)",
            a_path,
            b_path,
            len(a_factors),
            len(b_factors),
        )
    a_counter = Counter((ftype, keys) for ftype, keys, _ in a_factors)
    b_counter = Counter((ftype, keys) for ftype, keys, _ in b_factors)
    diff_a = a_counter - b_counter
    diff_b = b_counter - a_counter
    if diff_a or diff_b:
        logger.info("Structural diff for %s vs %s:", a_path, b_path)
        for label, diff in (("only_in_a", diff_a), ("only_in_b", diff_b)):
            if not diff:
                continue
            logger.info("  %s (showing up to 5):", label)
            for factor_sig, count in diff.most_common(5):
                factor_type, keys = factor_sig
                logger.info("    %s keys=%s count=%d", factor_type, keys, count)
        a_keys = {key for _, keys, _ in a_factors for key in keys}
        b_keys = {key for _, keys, _ in b_factors for key in keys}
        missing_in_b = sorted(a_keys - b_keys)
        missing_in_a = sorted(b_keys - a_keys)
        if missing_in_b:
            logger.info("  keys only in A (sample): %s", missing_in_b[:10])
        if missing_in_a:
            logger.info("  keys only in B (sample): %s", missing_in_a[:10])
        return False

    a_body_counter = Counter(body for _, _, body in a_factors)
    b_body_counter = Counter(body for _, _, body in b_factors)
    body_diff_a = a_body_counter - b_body_counter
    body_diff_b = b_body_counter - a_body_counter
    if body_diff_a or body_diff_b:
        logger.info("Measurement/value diff for %s vs %s:", a_path, b_path)
        for label, diff in (("only_in_a", body_diff_a), ("only_in_b", body_diff_b)):
            if not diff:
                continue
            logger.info("  %s (showing up to 3):", label)
            for body, count in diff.most_common(3):
                logger.info("    %s count=%d", body, count)
        return False

    logger.info("Match: %s vs %s (structure + values)", a_path, b_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True, help="Root directory with result subfolders.")
    parser.add_argument("--subdir_a", required=True, help="First subfolder name to compare.")
    parser.add_argument("--subdir_b", required=True, help="Second subfolder name to compare.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    graph_filenames = ["factor_graph.txt"]

    total = 0
    matched = 0
    missing = 0

    for base_dir in _iter_base_dirs(results_root):
        a_dir = base_dir / args.subdir_a / "metrics"
        b_dir = base_dir / args.subdir_b / "metrics"
        if not a_dir.exists() or not b_dir.exists():
            continue
        for filename in graph_filenames:
            a_path = a_dir / filename
            b_path = b_dir / filename
            if not a_path.exists() or not b_path.exists():
                missing += 1
                logger.info("Missing %s or %s", a_path, b_path)
                continue
            total += 1
            if _compare_file(a_path, b_path):
                matched += 1

    logger.info("Compared %d graphs: %d matched, %d mismatched, %d missing", total, matched, total - matched, missing)


if __name__ == "__main__":
    main()
