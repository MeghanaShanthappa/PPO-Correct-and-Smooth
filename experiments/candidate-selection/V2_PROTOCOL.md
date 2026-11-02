# V2: richer actions and lower-variance PPO

Frozen before V2 exports or controller runs. This is continued development on
base-model seed 0, not a new untouched benchmark. Preserve all V1 outputs.
The previously reported 7,865-node diagnostic cohort is excluded from the V2
data package and will not be scored or used to select changes in this run.
Official test targets are masked before any propagation. The base model used
the full official validation set for checkpoint selection, which remains a
limitation even when policy roles are separated.

## Hypotheses and changes

V1 had 215 training disagreements and only a binary raw/fixed-C&S choice.
V2 tests whether broader choices and graph context improve the controller.

Six frozen actions, all computed from the same original base probabilities:

| Action | Correction steps/alpha/scale | Smoothing steps/alpha |
|---|---|---|
| raw | none | none |
| fixed_cs | 1 / .45 / 20 | 1 / .6 |
| weak_cs | 1 / .2 / 10 | 1 / .3 |
| strong_cs | 1 / .45 / 20 | 3 / .8 |
| smooth_only | none | 1 / .6 |
| correct_only | 1 / .45 / 20 | none |

Use DAD normalization and the same disclosed node-order fix for every C&S
action. These are precomputed global propagation variants selected per node;
they do not implement simultaneous node-adaptive propagation or save the cost
of computing the candidate bank.

State: action confidence, margin, entropy, confidence/margin/entropy changes
relative to fixed C&S, prediction disagreement, raw predicted class one-hot,
and label-free degree/neighbor prediction statistics. No target labels,
correctness flags, node IDs, or true-class scores enter policy inputs.

Reward during training: correctness plus .05 * log(max(p_true,1e-8)). Labels
are used only on policy-training rows. The graded term provides signal where
actions predict the same class. Validation accuracy, not reward, selects the
checkpoint. Predictions are nonnegative and row-normalized before scoring.

All methods share a 32-hidden-unit tanh actor, seeds 20–24, Adam lr .001,
batch size 256 and gradient clipping .5. Twenty supervised warm-start epochs
precede the comparison. Equal best-reward actions share the target mass.

Compare: supervised warm-start alone; warm-start plus 60 further supervised
epochs; warm-start plus 60 expected-reward epochs (full-information control);
and warm-start plus 60 PPO epochs. Each continuation gets four update passes
per epoch. PPO samples one action per node per epoch, clips ratios at .2,
uses entropy coefficient .01, and centers rewards using their exact current
policy expectation over the available training action outcomes. This removes
the small learned critic; the baseline is independent of the sampled action.

Policy-training rows remain the original 23,593 positions; selection rows
remain the original 7,865 positions (permutation seed 20261028). Normalize
using only policy-training states. Initial fixed-C&S, warm-start and each
continuation checkpoint are eligible; selection-accuracy ties keep the earlier
checkpoint. Report all seeds, selection accuracy, fixed-C&S comparison, and
the best global action selected on that same selection set. The latter is a
development comparator, not an unbiased estimate. Save policy choices and
checkpoints so outputs can be independently recomputed.

No search over reward weights, architectures or action settings is included.
No diagnostic-holdout or official-test accuracy is produced. A stronger claim
requires a frozen subsequent evaluation across additional base-model seeds;
V2 selection gains alone are not paper-level evidence.

## Integrity amendment during preparation

The action-bank check identified six exact-score ties in the allowed
development cohort for which V1's `topk` prediction index differs from
`argmax`. V2's common `argmax` rule is retained for every candidate and learner.
The old exporter was fixed for future runs, and a regression test was added.
Original results remain archived. No protected diagnostic label was scored,
and no action settings or controller hyperparameters were changed in response.
