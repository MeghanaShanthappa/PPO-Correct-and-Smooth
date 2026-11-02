# Post-hoc diagnosis of the frozen MLP/C&S experiment

The strongest current explanation is that the learned action selector does not
generalize from training/validation to the official test population. This is
supported by measured differences, not a causal proof. No models were retrained,
no checkpoints were changed, and no new official-test accuracy evaluation was
performed for this diagnosis. All reported test accuracies come from the frozen
final evaluation. The diagnostic script reads saved choices and unlabeled graph
and prediction statistics; it never loads test targets.

## Integrity and implementation review

- All 155 artifacts recorded in the final lock still match their SHA256 hashes.
- CPU inference for base seed 0 / PPO actor 20 exactly reproduces saved validation
  choices, including feature normalization and action ordering.
- Seed-0 OOF target folds partition the official training nodes, with no overlap
  with validation. Code review confirms fold targets are excluded from C&S label
  anchors and from their MLP's training loss.
- Training and final scoring share action construction and state construction.
  Original C&S masks non-anchor targets internally in both paths. The PPO clipping
  expression matches the usual clipped probability-ratio objective.
- These checks found no obvious action-order, normalization, or split-index bug.
  They are not a complete independent numerical audit of every test prediction.
- All ten paired test differences are negative: -2.849 to -2.156 percentage points.

## Measured shift

Graph statistics below use the shared coalesced adjacency and official splits.
Confidence uses the seed-0 full MLP; accuracy averages ten base seeds.

| Quantity | Training | Validation | Test |
|---|---:|---:|---:|
| Mean degree | 171.49 | 92.44 | 39.02 |
| No directly connected training anchor | 0.50% | 2.14% | 26.04% |
| Mean fraction of neighbors in official training | 40.19% | 38.08% | 23.93% |
| Mean raw MLP confidence | 84.27% | 82.02% | 71.57% |
| Raw MLP accuracy | — | 83.607% | 64.320% |

The training graph statistics use all official training anchors; the OOF policy
actually sees only two-thirds of those anchors per fold. This table describes
the populations, not the exact per-fold anchor budget.

OGB intentionally uses product sales rank rather than a random split, making
transfer to less popular products a central challenge. See the
[official dataset description](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products).
The official files used here contain 196,615 training, 39,323 validation and
2,213,091 test nodes; use those actual counts rather than rounded documentation
percentages.

## Selector behavior

Fractions average all ten base models and five actor seeds per model.

| PPO action | Validation | Test |
|---|---:|---:|
| Raw MLP | 43.72% | 29.98% |
| Original fixed C&S | 6.74% | 6.34% |
| Correction scale 5 | 7.66% | 11.29% |
| Correction scale 20 | 12.64% | 21.72% |
| Smoothing alpha .6 | 26.87% | 23.76% |
| Smoothing alpha .9 | 2.37% | 6.91% |

The learned policy usually departs from the original baseline. Raw MLP is far
less accurate overall on test, yet still receives nearly 30% of choices, while
stronger correction receives more choices. These are plausible contributors;
aggregate action frequencies cannot establish which choices caused the loss.
That would require a separately labeled post-hoc error decomposition.

Both supervised selection (81.621% test) and the validation-best global action
(83.122%) lose to original C&S (84.207%). The failure therefore extends beyond
PPO's optimizer. PPO scores 81.740%, slightly above the supervised selector.

The OOF action accuracies resemble validation: original C&S averages 91.680%
on OOF training targets versus 91.500% on validation. For seed 0 / actor 20,
PPO gains 1.238 points on OOF targets, but this includes policy training examples
and is descriptive, not a held-out estimate.

Degree alone does not explain the failure. Seed-0 / actor-20 validation gains
are positive in every tested degree band: +1.857 points for degree <10,
+1.890 for 10–29, +0.726 for 30–99, and +0.688 for >=100. Low-degree validation
nodes are not an adequate stand-in for the full shifted test population.

## Design limitations and next experiment

Random OOF partitions preserve the popular training-node population. They
prevent direct target leakage but do not simulate sales-rank shift. All MLP
checkpoints, including fold models, and all controller checkpoints also use
official validation labels for selection; validation is not independent of
development. The reward combines accuracy with log probability, so calibration
may affect action preferences; its causal role remains untested. Current PPO is
a one-step contextual action selector over precomputed global C&S outputs,
not a policy learning sequential propagation steps.

A follow-up should first create train/validation-only stress splits that
separate anchor coverage and graph populations, rebuild OOF outputs with those
splits, and reserve a development holdout never used for checkpoint selection.
Compare fixed C&S, a small supervised selector, expected reward, and PPO with
identical inputs. Test a conservative fallback to original C&S under weak
support and measure gains in each development group. Degree matching alone is
insufficient given the positive low-degree validation results above.

Those proposals are hypotheses, not established fixes. The current Products
test set has been examined and must not become a tuning loop. Any newly tuned
method needs a fresh evaluation protocol or dataset for a credible confirmation.

Detailed machine-readable results:
`/var/tmp/mshanthappa/rl-cs/mlp/diagnosis/summary.json`.
Reproduction script: `diagnose_mlp.py`.
