#!/usr/bin/env bash
set -euo pipefail
cd /home/mshanthappa/ogbn-products-rlcs
exec >> /home/mshanthappa/ogbn-products-rlcs/mlp-report.log 2>&1
while tmux has-session -t rlcs-mlp-experiment 2>/dev/null; do
  /var/tmp/mshanthappa/rl-cs/grad-venv/bin/python summarize_mlp.py
  sleep 30
done
/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python summarize_mlp.py
