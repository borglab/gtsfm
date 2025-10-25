#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

# Pick the files that have staged or unstaged changes.
changed_files="$(
  git status --short \
  | awk '{print $2}' \
  | sort -u \
  | grep '\.py$' || true
)"

if [[ -z "${changed_files}" ]]; then
  echo "No modified Python files detected. Nothing to format."
  exit 0
fi

echo "Formatting:"
printf '  %s\n' ${changed_files}

black --line-length 120 ${changed_files}