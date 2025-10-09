#!/bin/bash
# =============================================================================
# Quick PR Comparison Script
# =============================================================================
# Usage: ./compare_prs.sh <master_pr> <branch_pr>
# =============================================================================

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <master_pr> <branch_pr>"
    echo "Example: $0 889 901"
    exit 1
fi

MASTER_PR="$1"
BRANCH_PR="$2"
OUTPUT_FILE="compare_${MASTER_PR}_${BRANCH_PR}.html"

# Run the comparison
python gtsfm/evaluation/visualize_benchmark_comparison.py \
  --master_path ~/Downloads/gtsfm/${MASTER_PR}/ \
  --branch_path ~/Downloads/gtsfm/${BRANCH_PR}/ \
  --output_path ${OUTPUT_FILE}

# Open the result
open ${OUTPUT_FILE}