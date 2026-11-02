# Reproducibility guide

## Two implementations

The existing GitHub root prototype (`rl_correct_smooth.py`) uses sequential REINFORCE actions. Its outputs must not be presented as the PPO candidate-selection results.

The completed PPO experiments are archived under `experiments/candidate-selection/` in the GitHub release. The local research checkout stores these scripts at its root. Their protocols and summaries are preserved with the archive.

## Recorded environment

Python 3.10; PyTorch 1.12.1+cu113; DGL 1.0.4; NumPy 1.23.5; SciPy 1.10.1; OGB 1.3.6. The full historical environment export is `requirements-grad.lock.txt`. CUDA-specific packages, GPU availability, and upstream source checkouts require separate setup; the root prototype's requirements are not sufficient for this archive.

The original source revisions and locations are in `upstream-lock.json` and `mlp-source-setup.json`. Obtain upstream repositories and datasets separately and preserve their licenses. Do not redistribute datasets or assume the archive includes trained weights or candidate banks.

## Read summaries without running training

`results/products-small-validation.json` preserves all three Products base seeds and controller comparisons. `results/arxiv-final-test.json` preserves the locked final Arxiv metrics. `results/products-original-test.json` preserves the earlier negative Products test result.

## Existing controller inputs

From the candidate-selection source directory, after preparing an input package as described by the protocols:

```bash
python train_cross_dataset.py --data /path/to/prepared-arxiv-seed \
  --output /path/to/NEW-policy-output --train-cap 32768
```

Products uses `train_v2.py` with the corresponding prepared Products input directory. These commands train controllers on existing candidate packages; they do not acquire data or reproduce the base MLP by themselves.

## Full reproduction limitations

The historical launchers and preparation scripts use absolute paths below `/var/tmp/mshanthappa/rl-cs` and `/home/mshanthappa/ogbn-products-rlcs`. Read and adapt those paths on a new machine before execution. `run_cross_dataset.sh` includes final test access, so it is not a development-only smoke test. Locked scorers intentionally reject repeat test access in completed run directories. Never delete lock markers or reuse completed output directories to bypass those guards.

The archive needs a portability pass before it can offer a one-command fresh-machine reproduction. Unit tests check small-scale invariants and leakage boundaries; they do not independently reproduce the full reported accuracies.
