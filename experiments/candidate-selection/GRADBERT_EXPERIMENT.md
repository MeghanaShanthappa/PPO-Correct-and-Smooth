# GraDBERT/GIANT + PPO-controlled C&S

The user selected the published GraDBERT+GIANT & SAGN+SLE+CnS Products
pipeline as the primary base. Published test accuracy is 86.92 ± 0.07%, and
validation accuracy is 93.71 ± 0.03%. These are reference numbers, not local
measurements. A target of +0.3 percentage points would correspond to 87.22%
test accuracy; it is an aspiration, not a promised result.

## Acquired assets

- Original GRAD revision `95f20ea94f7204fc4f813fc8239f4ba68c3956a3`.
- The author's Google Drive folder returned an empty file listing. Their
  [issue response](https://github.com/cmavro/GRAD/issues/1#issuecomment-1553526371)
  supplies an [Archive.org mirror](https://archive.org/download/gradbert_embeddings).
- `download_grad_features.py` retrieves only the Products `giant_feats.npy`
  and `gradbert_kd_feats.npy` members via HTTP ranges. Both passed ZIP CRC32,
  byte-count and NumPy shape checks; each is `[2449029, 768]`, float32.
- Feature SHA256 manifests are saved alongside the downloaded arrays at
  `/var/tmp/mshanthappa/rl-cs/features/gradbert-products`.
- Downloaded and processed the official OGB graph independently of the old
  text-bearing PyG cache: 2,449,029 nodes, 123,718,280 directed edge entries,
  47 classes and the official train/validation/test split.

## Baseline training

`run_grad_baseline.py` runs the authors' SAGN/SLE implementation with GIANT
features for stages 0, 1 and 2, then GraDBERT-KD features for stage 3. It uses
the released script's 100/100/100/10 epochs, hidden size 256, K=3,
label-K=9, thresholds .95/.95/.999, batch size 50,000 and learning rate .001.
The first run is base-model seed 0. Subsequent base-model seeds are separate
experiments; multiple actor seeds on one base model do not replace them.

Official test labels are read by the OGB dataset loader but immediately masked
to -1 before upstream preprocessing/training. The evaluation adapter scores
only the official validation set. It exports predictions for all graph nodes
because transductive self-labeling requires them. Exporting predictions for
test nodes does not expose their labels to training or reward construction.

Compatibility changes are recorded with source hashes and copied runtime code:

- CPU indexing into host-resident feature matrices when minibatch IDs are on
  CUDA; DGL copy_src/copy_edge aliases use their current equivalents.
- Separate GIANT and GraDBERT feature-cache keys so the final stage cannot
  reuse earlier-stage embeddings.
- Validation-only checkpoint selection/evaluation and saved selected weights.
- Isolated Python 3.10.22, torch 1.12.1+cu113, DGL 1.0.4, NumPy 1.23.5 and
  OGB 1.3.6. These differ from some original environment versions. A legacy
  torch library's executable-stack flag was cleared locally for this host;
  the reversible flag change is recorded in `grad-venv-execstack-fix.json`.

## C&S reference and a released-code defect

The supplied postprocessing script contains two different settings: its active
command uses one correction and one smoothing step, alphas .45/.6, scale 20;
its commented command uses 50/50 steps and different coefficients. We will
evaluate the active settings first and record the ambiguity. We cannot assume
the active command reproduces the historical leaderboard number exactly.

The non-autoscale correction implementation adds predictions in original node
order to residuals in train/validation/test concatenated order. The adapter
uses the same concatenated order for both terms. A test with a nonidentity
split order verifies this correction. This is a disclosed bug fix, not a
claimed PPO improvement. Both fixed C&S and PPO candidate outputs use the same
corrected implementation; PPO does not receive credit for fixing that bug.

## PPO comparison

First export `raw` and `fixed_cs` actions from the new baseline through
`postprocess_grad.py`. The previous LD/EmbeddingGemma exports are not used.
PPO chooses between these outputs per node; it does not apply an arbitrary
second C&S pass to already processed predictions.

Before viewing new controller results, freeze the same controller settings as
the initial pilot: seeds 20–24, split seed 20261028, 60/20/20 policy training /
selection / diagnostic holdout, 60 epochs, four update passes, hidden size 32,
learning rate .001, PPO clip .2 and entropy coefficient .01. Train a matched
supervised selector on the same available action outcomes.

Run the existing trainer with `--actions raw fixed_cs` and a new output path.
Report all actor seeds, both fixed baselines, action-selection ceiling,
introduced and corrected mistakes, and additional compute/parameters.

The base model and released features use validation-informed selection, so a
resplit of this validation set is exploratory. An official-test claim requires
freezing the pipeline before final evaluation, disclosing historical Products
test exposure, and paired comparisons across independent base-model seeds.
No repeated test scoring or test-driven tuning is part of this workflow.

## Evidence required to claim an improvement

1. Show actual base reproduction scores and deviations from the published
   implementation/settings. Do not compare a local validation score directly
   against the published test score.
2. Compare fixed C&S and PPO on the same base-model seed, features, candidate
   actions and evaluation nodes; include the supervised selector.
3. Report the paired accuracy differences across multiple base-model seeds,
   not only the best run. A positive exploratory point estimate is insufficient
   to establish a robust improvement or an RL-specific advantage.
4. Preserve negative/zero results. If PPO does not help, report that outcome.

The first seed-0 workflow is now complete after recovering a launcher error
without retraining. See [RESULTS_GRAD_SEED0.md](RESULTS_GRAD_SEED0.md) for the
matched comparison. PPO averaged +0.0127 percentage points over fixed C&S on
the diagnostic subset, while raw predictions scored higher than both. This
does not establish a robust PPO benefit.
