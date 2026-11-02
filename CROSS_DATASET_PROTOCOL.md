# Frozen cross-dataset follow-up: ogbn-products and ogbn-arxiv

Goal: test whether the small-adjustment C&S/PPO method helps on a second OGB
dataset. Report negative outcomes. No claim of improvement is made in advance.
This tests independently trained controllers on each dataset, not zero-shot
transfer of Products controller weights. Three base seeds (0,1,2), five actor
seeds (20–24) each. Products results are already known and informed this design;
arxiv test accuracy has not been used for this development protocol.

## Dataset and baseline

Use ogbn-arxiv's official time split: 90,941 train, 29,799 validation, 48,603
test nodes, 40 classes. Source archive:
https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
See https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv and
https://ogb.stanford.edu/docs/leader_rules/.

Binary, symmetric, coalesced citation adjacency without added self-loops.
128 official node features, standardized with label-free transductive statistics.
No external data or extra spectral/diffusion features. The Products reference
uses its existing 100+128 spectral features; disclose the feature difference.
This is a common-controller experiment, not a reproduction of the stronger
published arxiv C&S configuration or a claim to beat its leaderboard entry.

Original upstream MLP class: 3 linear layers, hidden width 200, BN then ReLU
(the upstream arxiv convention), dropout .5, Adam .01, 299 epochs. Checkpoint
using official validation accuracy. Save full and three training-fold models.
Fold permutation seed 20261029; each target fold excluded from its own MLP
training loss and C&S anchors. Other fold checkpoints also use validation for
selection. Test labels are masked immediately after reading dataset arrays.

## Controller held fixed

Five actions identical to the Products small-adjustment experiment: scale9,
fixed_cs (scale10), scale11, all smoothing alpha .8; smooth075 and smooth085,
both scale10. All use correction alpha1=1, 50/50 iterations, DAD/DA operators.
Fixed C&S is action index 1. Genuine raw predicted class, per-action confidence,
margin/entropy, differences from baseline, disagreement, and the same five
structural features. Raw class one-hot shrinks from 47 to 40 dimensions, so
controller input width is 80; hidden width remains 32.

All PPO rewards and SFT/expected-reward targets use OOF official training
labels only. No validation targets used in gradient updates or propagation.
Reward correctness + .05 log true-class probability. Same training cap32768,
SFT warm20/continuation60 epochs, four passes, batch256, Adam.001, clip.2,
entropy.01, gradient clip.5. Five actor seeds, matched controls. Source copy
train_cross_dataset.py changes input-boundary checks only from train_v2.py.

## Final evaluation and reporting

Complete all three development runs without changing this protocol. Freeze
code, actor checkpoints, normalization, predictions, graph, dataset cache,
and validation-best fixed actions in a hash lock before accessing test labels.
Then perform one final arxiv evaluation of fixed C&S, PPO, supervised,
expected-reward, and validation-selected fixed action. Report mean and sample
standard deviation across three base seeds, averaging actors within each base.
No tuning or reruns after test access. No new Products test evaluation is
included; the earlier negative Products test result is retained and disclosed.

No external publication, submission or leaderboard claim is authorized by
this local experiment. Results appear in RESULTS_CROSS_DATASET.md.
