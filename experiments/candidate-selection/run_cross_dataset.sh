#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/cross-dataset.log 2>&1
cd /home/mshanthappa/ogbn-products-rlcs
source env_mlp.sh
storage=/var/tmp/mshanthappa/rl-cs/cross-dataset-v1/ogbn-arxiv
python_bin=/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python
mkdir -p "$storage"
trap 'printf "FAILED at line %s\n" "$LINENO" > "$storage/FAILURE"' ERR
for seed in 0 1 2; do
  "$python_bin" -u cross_dataset.py --phase prepare --seed "$seed"
  if [[ ! -f "$storage/policy-$seed/summary.json" ]]; then
    "$python_bin" -u train_cross_dataset.py --data "$storage/seed-$seed" --output "$storage/policy-$seed" --train-cap 32768
  fi
  "$python_bin" report_cross_dataset.py
done
printf 'All three arxiv development runs completed.\n' > "$storage/DEVELOPMENT_COMPLETE"
"$python_bin" -u cross_dataset.py --phase final
"$python_bin" report_cross_dataset.py
printf 'Cross-dataset arxiv experiment and locked final scoring complete.\n' > "$storage/COMPLETE"
