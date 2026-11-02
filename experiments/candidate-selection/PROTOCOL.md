# OGB-Products: RL-controlled Correct and Smooth

This is a new local project. Its objective is to test whether an RL controller
improves fixed base-model predictions beyond standard validation-tuned C&S and
a supervised controller using the same information. Improvement is a research
hypothesis, not an established result.

## Reference models requested

The official OGB table reports test accuracy followed by validation accuracy:

| Pipeline | Published test | Published validation | Code |
|---|---:|---:|---|
| LD+GIANT+SAGN+SCR | 87.18 ± 0.04% | 93.99 ± 0.02% | https://github.com/MIRALab-USTC/LD |
| GraDBERT+GIANT & SAGN+SLE+CnS | 86.92 ± 0.07% | 93.71 ± 0.03% | https://github.com/cmavro/GRAD |
| GIANT-XRT+R-SAGN+SCR+C&S | 86.84 ± 0.05% | 93.65 ± 0.03% | https://github.com/Eigenworld/RandomAttention/tree/main/R-SAGN |
| GIANT-XRT+SAGN+SCR+C&S | 86.80 ± 0.07% | 93.57 ± 0.04% | https://github.com/THUDM/SCR |

Source: https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products.
These are published references, not reproduced results. These pipelines combine
feature learning, graph classification, multi-stage training and sometimes
post-processing; they are not interchangeable pretrained checkpoints. Do not
add a second C&S pass to an already post-processed prediction without an
explicitly labeled experiment. Preserve the exact feature and stage settings.

The user subsequently selected **GraDBERT+GIANT & SAGN+SLE+CnS** as the
primary reproduction target. LD/SAGN is retained as a reference and its
historical pilot is not evidence for GraDBERT.
Downloading source does not establish that weights, feature files or original
training artifacts are available.

## First milestone: cached-action PPO pilot

A prior local project has train/validation-only exports for raw and C&S
predictions from a ten-seed ensemble of LD + frozen EmbeddingGemma walk-text
features + raw features, followed by SAGN/SLE. It also has an earlier adaptive
contextual-bandit experiment and historical test use. Those files are preserved.
This project does not rerun their official-test evaluator.

The cached stack is **not an exact reproduction of any of the four reference
models**. This initial pilot validates the controller implementation and
compares learning objectives. It cannot establish a new leaderboard score.

The pilot consumes only 39,323 official-validation rows from exports containing
196,615 training rows and zero test rows. It does not load the graph dataset
or any official test predictions/labels. Training rows are not used for
controller learning because their base predictions are in-sample.

Before the first run, freeze these settings:

- Index-only permutation seed: 20261028. Split 60% policy training, 20%
  checkpoint selection, 20% diagnostic holdout. Save the exact row positions.
- Actor seeds: 20, 21, 22, 23, 24. A 13-input, 32-hidden-unit tanh actor chooses
  raw prediction or the existing fixed C&S prediction.
- Inputs: confidence, margin, entropy from both actions; signed/absolute
  differences; prediction disagreement. No target label or node identifier.
- Fit normalization only on policy-training rows. Both learners train on the
  same rows where action predictions disagree; all held-out rows are scored.
- PPO: one-step terminal episodes, sampled categorical actions, clipped
  objective with epsilon 0.2, entropy coefficient 0.01, learned value baseline.
  Reward is sampled-action correctness minus fixed-C&S correctness.
- Supervised control: same actor; cross-entropy on training rows where only
  one action is correct. Same epoch and update-pass budgets. PPO also trains a
  critic, so wall-clock cost and parameter overhead must be reported.
- 60 epochs, four update passes per epoch, batch size 256, Adam learning rate
  0.001, gradient clipping 0.5. Checkpoint selection uses only the middle split;
  ties keep the earlier checkpoint. Do not tune from diagnostic results.
- Report all five seeds, fixed raw and C&S baselines, oracle action-selection
  ceiling, corrected mistakes and introduced mistakes. The oracle is an upper
  bound using evaluation labels, never a deployable policy.

This is PPO for a contextual bandit, not sequential graph editing, an LLM
fine-tune, or adaptive propagation inside C&S. It chooses cached action outputs
and therefore does not save the cost of computing those outputs. DPO is not
implemented in this milestone.

The entire official validation set and its base-model configuration were used
historically. A new partition does not create untouched data. All pilot results
must be labeled exploratory; same-graph dependencies also prevent treating
nodes as independent observations. Seed variation here concerns controller
training, not five independent base-model reproductions.

## Subsequent experiments

1. Reproduce the selected GraDBERT/GIANT baseline using released features and original
   settings; record pre-C&S predictions, source revisions and feature hashes.
2. For a stronger protocol, create controller-training folds inside the
   official training nodes. For each fold, exclude its labels from base-model
   fitting, C&S anchors and self-label stages before making reward targets.
   Supervised feature extraction such as LD must respect the same exclusion;
   a feature tensor trained on all target labels is not out-of-fold evidence.
3. Compare fixed C&S, a validation-tuned global configuration, simple confidence
   rules, supervised selection, full-information expected-reward optimization,
   and PPO using matched action sets and tuning budgets.
4. Expand actions to explicit C&S strengths or stopping decisions. Recompute
   propagation under the chosen action semantics; mixing final predictions is
   not equivalent to changing propagation at intermediate layers.
5. Use validation for selection; record a frozen protocol before final test
   evaluation. Disclose earlier official-test exposure. Broaden to another
   graph to support a paper rather than relying on one Products result.

No checkpoints, predictions, paper claims or scores may be selected using
official test feedback during development.
