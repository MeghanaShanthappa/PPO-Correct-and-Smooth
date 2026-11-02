#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/small-cs.log 2>&1
cd /home/mshanthappa/ogbn-products-rlcs
source env_mlp.sh
storage=/var/tmp/mshanthappa/rl-cs/small-cs-v1
python_bin=/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python
mkdir -p "$storage"
trap 'printf "FAILED at line %s\n" "$LINENO" > "$storage/FAILURE"' ERR
for seed in 0 1 2; do
  "$python_bin" -u prepare_small_cs.py --seed "$seed" --output "$storage/seed-$seed"
  if [[ ! -f "$storage/policy-$seed/summary.json" ]]; then
    "$python_bin" -u train_v2.py --data "$storage/seed-$seed" --output "$storage/policy-$seed" --train-cap 32768
  fi
  "$python_bin" summarize_small_cs.py
done
printf 'Three-seed development experiment complete. No new test evaluation.\n' > "$storage/COMPLETE"
