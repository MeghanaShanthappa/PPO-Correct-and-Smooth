# Original MLP + C&S reproduction and PPO comparison

Primary source: https://github.com/Chillee/CorrectAndSmooth, revision
`b910314a59270984f5e249462ee3faa815fc9a0c`.
Published Products reference: test 84.18 ± 0.07%, validation 91.47 ± 0.09%.
Those are different splits; neither is a result of the new experiment yet.

## Exact feature/model target

The requested 96,247-parameter MLP uses the official 100 OGB features and
128 label-free spectral features. No GIANT, GraDBERT, LD or external text
features are included. The upstream model has three linear layers, hidden
width 200, ReLU then batch normalization, dropout .5, and 47 output classes.
Adam lr .01; 299 epochs, matching `range(1,300)` in the released code.
Select checkpoints using official validation accuracy. Test labels are masked
before training; raw/corrected model outputs are saved without test scoring.
Ten base-model seeds 0–9 are planned; their standard deviation is reported
only when all ten complete, never substituted with actor-seed variation.

The original spectral operator is
`L = I + D (A + tau/N * 11^T) D`, where tau is mean degree and
`D_ii = 1/sqrt(degree_i+tau)`. Compute its 128 largest-magnitude eigenvectors
at tolerance 1e-6, then apply the source's pivoted-QR/SVD SCDM rotation.
The reproduction ports the Julia/ARPACK operator to a GPU sparse eigensolver;
it is mathematically matched, not byte-identical Julia output. Save residuals,
orthogonality, revisions and numerical runtime details. Validate against a
small dense reference before processing Products. Graph adjacency is binary,
undirected and coalesced, without added self-loops, matching the source.

## Original fixed C&S

Use the upstream functions, not the GraDBERT C&S implementation:
50 correction steps, alpha1=1, DAD normalization, fixed scale 10;
50 smoothing steps, alpha2=.8, DA normalization; training labels only.
The alpha1=1 error update reanchors training residuals on every iteration.
Normalized sparse matrix multiplication uses GPU kernels; test equivalence
against dense CPU multiplication on a small graph.
The original functions infer the number of classes from all labels. Because
non-anchor labels are masked, use the public output dimension (47) instead;
otherwise training anchors alone have a smaller maximum class index. This
preserves the original 47-column operation without reading held-out labels.

## PPO action bank

Freeze six candidates before results: raw, original fixed C&S, correction
scale 5 + smoothing .8, correction scale 20 + smoothing .8, correction scale
10 + smoothing .6, and correction scale 10 + smoothing .9. Every candidate
uses 50/50 steps and the same DAD/DA normalizations. These are precomputed
global candidate outputs selected per node, not a second C&S pass and not
simultaneously adaptive propagation.

Reuse V2's 94-feature label-free state and its frozen PPO/SFT/full-information
comparison: actor seeds 20–24, hidden width 32, 20 warm-start and 60 continuation
epochs, four update passes, lr .001, batch 256, clip .2, entropy .01; training
reward correctness + .05 log true-class probability. Report all actor seeds.
Before MLP or policy results, strengthen the label-budget protocol: create
three index-only folds of the 196,615 official training nodes (seed 20261029).
For each fold, train a separate MLP on the other training nodes and exclude
the target fold from C&S anchors. Construct controller inputs/rewards only
from that fold's out-of-fold outputs. Spectral features are unsupervised and
shared. Train the controller on an index-only sample of at most 32,768 OOF
training examples per actor seed, shared by all compared objectives. This
keeps policy optimization bounded while providing far more examples than V1.
Use the whole official validation set for common checkpoint selection and the
matched MLP/C&S/PPO validation comparison. Its prior use is disclosed; it is
not an untouched test set. No validation labels enter controller rewards or
C&S anchors. Freeze actor/checkpoint choices before any new official-test
evaluation. Selection improvements are not held-out proof.

Store each base seed separately. Compare PPO with original fixed C&S, raw,
the selection-best global candidate, and matched supervised/expected-reward
controllers. Full-validation C&S reproduction metrics are separate from
policy selection scores. Do not compare the latter directly to 91.47%.

The published test reference and previous Products test exposure are already
known. Any later test evaluation must disclose this history, freeze both
methods beforehand, and avoid test-driven adjustment. Initial reproduction
and development comparisons do not score official test labels.

After all ten development runs complete, freeze source and actor/prediction
hashes in a durable lock, write TEST_ACCESS_STARTED before loading targets,
and score original fixed C&S versus frozen PPO actors once on the same
2,213,091 official-test nodes for each base seed. No optimization occurs in
that phase. Keep ten paired base-model differences and actor-level scores.
Freeze the validation-best global action before test access and report its
test accuracy, along with the matched supervised selector, as controls.
The final scorer refuses a second invocation after test access starts. The
known prior Products test exposure remains disclosed; no result is a reason
to adjust and repeatedly score the test set. This final phase completes the
requested comparison to the published test reference.
