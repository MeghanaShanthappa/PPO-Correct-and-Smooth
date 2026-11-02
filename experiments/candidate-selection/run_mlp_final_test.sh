#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/mlp-final-test.log 2>&1
cd /home/mshanthappa/ogbn-products-rlcs
source env_mlp.sh
root=/var/tmp/mshanthappa/rl-cs/mlp
while [[ ! -f "$root/COMPLETE" ]]; do
  if ! tmux has-session -t rlcs-mlp-experiment 2>/dev/null; then
    printf 'Development workflow stopped; final test not accessed.\n'
    exit 1
  fi
  sleep 30
done
/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python -u evaluate_mlp_test_once.py
