# V2 PPO-controlled C&S: development results

This run uses one GraDBERT/GIANT base-model seed and five controller seeds. All numbers below are accuracy on the 7,865-node **selection set**, used to select checkpoints. They are not held-out accuracy or official test results. The earlier diagnostic cohort was excluded from the data package.

## Fixed action bank

| Action | Selection accuracy |
|---|---:|
| raw | 93.2104% |
| fixed_cs | 93.4011% |
| weak_cs | 93.2104% |
| strong_cs | 92.8417% |
| smooth_only | 93.2231% |
| correct_only | 93.2359% |

## Controller comparison

| Controller | Mean selection accuracy | Gain over fixed C&S (pp) | Gain over best global action (pp) |
|---|---:|---:|---:|
| warm_start | 93.4240% | +0.0229 | +0.0229 |
| supervised | 93.4342% | +0.0331 | +0.0331 |
| expected_reward | 93.4240% | +0.0229 | +0.0229 |
| ppo | 93.4622% | +0.0610 | +0.0610 |

PPO minus the supervised warm-start mean: **+0.0381 percentage points**.

The best global action on this same selection set is `fixed_cs`. Its selection advantage is not an independent estimate either. Gains over the original fixed C&S setting alone cannot establish that PPO is responsible; compare both global action tuning and supervised training.

## What changed

The controller has six precomputed actions, 94 label-free state features, a 32-unit hidden layer (3,238 actor parameters), supervised initialization, and a dense training reward combining correctness and true-class log probability. PPO uses an exact action-independent baseline computed from training action outcomes. The same architecture and data support the supervised and expected-reward controls.

The original exporter used `topk` to choose the predicted class. Six allowed development rows had exact tied top scores and differed from standard `argmax`. V2 consistently uses `argmax` across every method; the old results are preserved with a caveat. No PPO credit should be assigned to changing tie-breaking.

## Limits and verification

All 20 saved choice vectors were checked against the reported accuracies and corrected/introduced mistake counts. The development data hash matches its training configuration. Eleven code tests passed before the experiment completed.

The base model used full-validation checkpoint selection. These development results therefore do not establish generalization, a robust PPO improvement, a leaderboard score, or a paper-level result. Future evaluation must freeze the method and include additional base-model seeds and appropriately isolated targets. All six global propagation candidates must be computed before selection; this implementation does not reduce propagation cost.

Raw run records: `/var/tmp/mshanthappa/rl-cs/runs/grad-v2-policies`. The frozen protocol is [V2_PROTOCOL.md](V2_PROTOCOL.md).
