#!/usr/bin/env bash
set -euo pipefail
exec >> /home/mshanthappa/ogbn-products-rlcs/grad-experiment.log 2>&1
export DGLBACKEND=pytorch
export MPLBACKEND=Agg
export MPLCONFIGDIR=/var/tmp/mshanthappa/rl-cs/matplotlib
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
project=/home/mshanthappa/ogbn-products-rlcs
storage=/var/tmp/mshanthappa/rl-cs
python_bin="$storage/grad-venv/bin/python"
cd "$project"
trap 'printf "FAILED at line %s\n" "$LINENO" > "$storage/runs/grad-experiment-failure.txt"' ERR

# Transfer the initial interactive invocation to this persistent workflow only
# after it has finished writing its reusable feature cache, or has exited.
if [[ -n "${TRANSFER_BASELINE_PID:-}" ]]; then
  "$python_bin" -u - "$TRANSFER_BASELINE_PID" <<'PY'
import os,signal,sys,time,zipfile
from pathlib import Path
pid=int(sys.argv[1])
cache=Path('/var/tmp/mshanthappa/rl-cs/ogb/ogbn_products/embedding/smoothed_embs_K_3_giant.pt')
for _ in range(180):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        alive=b'run_grad_baseline.py' in cmd and b'grad-baseline-seed0' in cmd
    except FileNotFoundError:
        alive=False
    complete=cache.exists() and zipfile.is_zipfile(cache)
    if complete or not alive:
        if alive:
            os.kill(pid,signal.SIGINT)
            print('Initial command stopped after cache completion; fresh persistent baseline follows.',flush=True)
            for _ in range(60):
                if not Path(f'/proc/{pid}').exists():
                    break
                time.sleep(1)
            else:
                raise RuntimeError('initial invocation did not stop; refuse concurrent baseline')
        if cache.exists() and not complete:
            cache.rename(cache.with_suffix('.incomplete-initial-attempt'))
        break
    print(f'Waiting for initial cache: {cache.stat().st_size if cache.exists() else 0} bytes',flush=True)
    time.sleep(10)
else:
    raise RuntimeError('transfer did not finish; refuse concurrent baseline')
PY
fi

if [[ ! -f "$storage/runs/grad-seed0-main/BASE_TRAINING_COMPLETE" ]]; then
"$python_bin" -u run_grad_baseline.py \
  --upstream "$storage/upstream/GRAD" \
  --giant "$storage/features/gradbert-products/giant_feats.npy" \
  --gradbert "$storage/features/gradbert-products/gradbert_kd_feats.npy" \
  --data-root "$storage/ogb" \
  --output "$storage/runs/grad-seed0-main"
fi

if [[ ! -f "$storage/runs/grad-seed0-actions/manifest.json" ]]; then
"$python_bin" -u postprocess_grad.py \
  --baseline-run "$storage/runs/grad-seed0-main" \
  --data-root "$storage/ogb" \
  --output "$storage/runs/grad-seed0-actions"
fi

if [[ ! -f "$storage/runs/grad-seed0-ppo/summary.json" ]]; then
"$python_bin" -u train_pilot.py \
  --candidates "$storage/runs/grad-seed0-actions" \
  --actions raw fixed_cs \
  --output "$storage/runs/grad-seed0-ppo"
fi

printf 'COMPLETED: seed-0 exploratory baseline/C&S/PPO comparison; no official test scoring.\n' > "$storage/runs/grad-experiment-complete.txt"
