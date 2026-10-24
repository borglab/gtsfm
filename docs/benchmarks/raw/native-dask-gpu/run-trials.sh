#!/usr/bin/env bash
# Execute only after the coordinator's explicit SOURCE_FREEZE_AND_GPU_GRANTED.
set -euo pipefail
base=/home/jasdeep/gtsfm-native-dask-20261024
export PATH=/home/jasdeep/gtsfm_eval/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib
test -f "$base/SOURCE_FREEZE_AND_GPU_GRANTED.json"
for definition in 'A1 manual' 'B1 native' 'A2 manual' 'B2 native'; do
  read -r trial arm <<< "$definition"
  unit="native-dask-${trial,,}"
  test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
  set +e
  systemd-run --user --wait --collect --unit="$unit" \
    -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=400% -p RuntimeMaxSec=960 \
    --setenv=PATH="$PATH" --setenv=PYTHONDONTWRITEBYTECODE=1 \
    /home/jasdeep/gtsfm_eval/.venv/bin/python "$base/experiment_dask_configuration_gpu.py" \
    --base "$base" --trial "$trial" --arm "$arm" --dataset "$base/dataset-first4" --wall-seconds 900 \
    > "$base/evidence/$trial-systemd.log" 2>&1
  status=$?
  set -e
  journalctl --user -u "$unit" --no-pager -o cat > "$base/evidence/$trial-journal.log"
  printf '%s\n' "$status" > "$base/evidence/$trial-systemd-exit.txt"
  cat "$base/evidence/$trial-systemd.log"
  systemctl --user show "$unit" -p LoadState -p ActiveState -p Result > "$base/evidence/$trial-unit-after.txt" || true
  if test "$status" -ne 0; then
    printf 'Trial %s failed; series stopped, retaining all artifacts.\n' "$trial"
    exit "$status"
  fi
  test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
  printf 'Trial %s finished and GPU compute list is empty.\n' "$trial"
done
