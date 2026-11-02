#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/mlp-spectral.log 2>&1
cd /home/mshanthappa/ogbn-products-rlcs
source env_mlp.sh
/var/tmp/mshanthappa/rl-cs/grad-venv/bin/python -u prepare_mlp_spectral.py \
  --data-root /var/tmp/mshanthappa/rl-cs/ogb \
  --output /var/tmp/mshanthappa/rl-cs/mlp/spectral
