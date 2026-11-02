#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/mlp-experiment.log 2>&1
cd /home/mshanthappa/ogbn-products-rlcs
source env_mlp.sh
storage=/var/tmp/mshanthappa/rl-cs/mlp
python_bin=/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python
mkdir -p "$storage"
trap 'printf "FAILED at line %s\n" "$LINENO" > "$storage/FAILURE"' ERR
while [[ ! -f "$storage/spectral/manifest.json" ]]; do
  if ! tmux has-session -t rlcs-mlp-spectral 2>/dev/null; then
    printf 'Spectral preprocessing did not complete; inspect mlp-spectral.log.\n' > "$storage/FAILURE"
    exit 1
  fi
  sleep 30
done
for seed in {0..9}; do
  printf '\nStarting MLP base seed %s\n' "$seed"
  "$python_bin" -u run_mlp_seed.py \
    --spectral "$storage/spectral" \
    --data-root /var/tmp/mshanthappa/rl-cs/ogb \
    --seed "$seed" --output "$storage/seed-$seed"
  if [[ ! -f "$storage/policy-$seed/summary.json" ]]; then
    "$python_bin" -u train_v2.py --data "$storage/seed-$seed" \
      --output "$storage/policy-$seed" --train-cap 32768
  fi
  printf 'Completed base/policy seed %s\n' "$seed"
done
printf 'All ten MLP baseline/OOF/PPO development comparisons complete; no official test score.\n' > "$storage/COMPLETE"
