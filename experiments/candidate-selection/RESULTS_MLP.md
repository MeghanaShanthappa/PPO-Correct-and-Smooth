# Original MLP + C&S and PPO reproduction

Completed baseline scores: 10/10. Completed paired PPO comparisons: 10/10.

Every local score below uses the **full 39,323-node official validation set**. That set selects checkpoints, so these are development comparisons, not independent test results. The published reference is 91.47% validation and 84.18% test; the latter has not been reproduced or scored in this run.

| Base seed | Raw MLP validation | Original C&S validation | PPO mean validation | PPO minus C&S (pp) |
|---|---:|---:|---:|---:|
| 0 | 83.5135% | 91.5190% | 92.4660% | +0.9470 |
| 1 | 83.4423% | 91.5952% | 92.4680% | +0.8728 |
| 2 | 83.7017% | 91.4401% | 92.4324% | +0.9923 |
| 3 | 83.6025% | 91.4528% | 92.3582% | +0.9053 |
| 4 | 83.5236% | 91.5596% | 92.3490% | +0.7894 |
| 5 | 83.6711% | 91.4935% | 92.4884% | +0.9948 |
| 6 | 83.8415% | 91.4401% | 92.4467% | +1.0065 |
| 7 | 83.4957% | 91.5317% | 92.3149% | +0.7833 |
| 8 | 83.6533% | 91.6410% | 92.5377% | +0.8967 |
| 9 | 83.6254% | 91.3257% | 92.4594% | +1.1337 |

The model is the original 96,247-parameter architecture with default OGB features and graph-only spectral features. The spectral eigensolver is a disclosed GPU port of the original Julia operator with residual/orthogonality checks. C&S uses the original 50/50-step DAD/DA implementation.

PPO training rewards come from three-fold out-of-fold predictions of official training nodes. Each target fold is excluded from classifier training and C&S anchors. No validation labels are used in policy rewards. The same training label budget is available to the fixed baseline and policy. Validation labels still select base and policy checkpoints.

Five actor seeds are averaged within each base-model run; they do not replace independent base-model seeds. Report all ten base runs before claiming the published mean and standard deviation are reproduced. No official test improvement is established by validation results alone.

Frozen setup: [MLP_PROTOCOL.md](MLP_PROTOCOL.md). Live log: [mlp-experiment.log](mlp-experiment.log).
