# RL-controlled Correct and Smooth on OGB-Products

Local research project testing whether PPO can improve a fixed graph
classifier through selective use of Correct and Smooth (C&S).

## Current work: original MLP + C&S reproduction

The user selected the original **96,247-parameter MLP + C&S** baseline for the
next comparison. It uses default OGB features plus 128 graph-only spectral
features, with the original 50/50-step correction/smoothing settings. The
first base seed achieved **91.5190% full-validation accuracy**, compared with
the published 91.47 ± 0.09% reference. This is one run, not a ten-run aggregate
or an official-test reproduction yet.

Ten independent base seeds are running. PPO training examples come from
three-fold out-of-fold predictions of official training nodes, excluding each
target fold from MLP fitting and C&S anchors. Fixed C&S, validation-selected
global C&S, PPO and supervised selection will be compared on matched nodes.
Final official-test scoring is queued only after all development artifacts
are frozen; it has not begun.

See [MLP_PROTOCOL.md](MLP_PROTOCOL.md), the updating
[validation report](RESULTS_MLP.md), and [live log](mlp-experiment.log).
The spectral GPU port passed eigen-residual and orthogonality checks. Sixteen
tests passed, including the parameter count, original C&S equations and
held-out-label exclusion. Earlier GraDBERT experiments remain preserved below.

## Latest: V2 development experiment completed

V2 adds six candidate processing strengths, label-free neighborhood context,
supervised initialization, and a lower-variance PPO baseline. On the 7,865-node
**selection set**, fixed C&S scored 93.4011%; PPO averaged 93.4622% across
five actor seeds (+0.0610 percentage points). The supervised continuation
averaged 93.4342%. These are checkpoint-selection results, not held-out or
official-test evidence. The previously viewed diagnostic cohort was excluded.
Eleven tests passed and all 20 saved policy-choice vectors were independently
recomputed against reported metrics. See [RESULTS_V2.md](RESULTS_V2.md) and
the frozen [V2_PROTOCOL.md](V2_PROTOCOL.md).

V2 lives in `prepare_v2.py`, `train_v2.py`, and `report_v2.py`. Its local
data and results are under `/var/tmp/mshanthappa/rl-cs/runs/grad-v2-development`
and `/var/tmp/mshanthappa/rl-cs/runs/grad-v2-policies`. The original V1 runs
remain preserved, including a documented tie-breaking caveat.

The selected primary baseline is now **GraDBERT+GIANT & SAGN+SLE+CnS**.
Both author-released Products feature files and the official OGB graph have
been acquired. The first validation-only base-training and PPO comparison
are complete. On the matched 7,865-node diagnostic subset, fixed C&S scored
94.0496%, PPO averaged 94.0623%, and raw predictions scored 94.2022%.
The PPO increase over C&S is one net correct node on average across five actor
seeds; it does not establish a robust improvement. See
[completed results](RESULTS_GRAD_SEED0.md).
See [GRADBERT_EXPERIMENT.md](GRADBERT_EXPERIMENT.md) for the exact settings,
compatibility changes, C&S reference ambiguity and requirements for an
improvement claim. Only one base-model seed has been evaluated so far.

The requested reference pipelines are LD+GIANT+SAGN+SCR,
GraDBERT+GIANT & SAGN+SLE+CnS, GIANT-XRT+R-SAGN+SCR+C&S, and
GIANT-XRT+SAGN+SCR+C&S. Their released source has been downloaded and pinned
in `upstream-lock.json`. Exact baseline reproduction is not yet complete.

Read [PROTOCOL.md](PROTOCOL.md) for reference scores, experimental boundaries,
the frozen first-pilot configuration, and the next reproduction stages.

## First completed milestone

A real PPO implementation now chooses between cached raw and C&S outputs. It
uses sampled actions, clipped importance ratios, entropy regularization and a
value baseline. A matched-actor supervised learner provides a simpler control.

This first run uses historical validation-only exports from an LD-based
**hybrid stack with additional EmbeddingGemma features**, rather than an exact
reproduction of the four reference pipelines. It reuses a historically tuned
validation pool. It is an exploratory controller pilot, not a new official
test score or evidence of state-of-the-art improvement.

| Method | Diagnostic subset accuracy |
|---|---:|
| Raw predictions | 94.215% |
| Fixed C&S | 94.520% |
| Supervised selector, five actor seeds | 94.556% |
| PPO selector, five actor seeds | 94.558% |

The diagnostic subset has 7,865 nodes. PPO improves over fixed C&S by three
net predictions per seed (+0.038 percentage points), while the supervised
selector averages 2.8 net predictions. This does not establish an advantage
for PPO. Repeated actor seeds share the same base ensemble and evaluation
nodes; they are not independent baseline reproductions.

## Run the pilot

The small controller requires Python, PyTorch and NumPy. The current host has
a compatible pre-existing environment at
`/var/tmp/mshanthappa-robotics/venv/bin/python` (PyTorch 2.11.0+cu130,
NumPy 2.4.6). This is a reused runtime, not a newly installed isolated graph
training environment. The pilot runs on CPU and does not need DGL or the full
graph in memory.

```bash
cd /home/mshanthappa/ogbn-products-rlcs
/var/tmp/mshanthappa-robotics/venv/bin/python -m unittest discover -s tests -v
/var/tmp/mshanthappa-robotics/venv/bin/python train_pilot.py \
  --candidates /home/mshanthappa/ogbn-products-leaderboard/results/rl_v3/candidate_predictions \
  --output /var/tmp/mshanthappa/rl-cs/runs/NEW-RUN-NAME
```

Use a new output directory: existing runs are never overwritten. `--help`
lists all configuration arguments. The script validates zero-test-row metadata
and expected array dimensions, and consumes only official-validation rows.
It does not open the graph file, full base predictions or official test labels.

## Artifacts

- `train_pilot.py`: controller training, checkpoint selection and diagnostics.
- `run_grad_baseline.py`: original-source GraDBERT/GIANT staged training with
  validation-only evaluation and test-label masking.
- `postprocess_grad.py`: fixed C&S and raw action exports for the selected base.
- `download_grad_features.py`: selective, resumable author-mirror downloads.
- `tests/test_pilot.py`: PPO clipping, split isolation, label-blind holdout and
  rejection of test-containing exports.
- `upstream-lock.json`: baseline source URLs, revisions and local locations.
- `BASELINE_READINESS.md`: exact reproduction inventory and remaining inputs.
- `/var/tmp/mshanthappa/rl-cs/runs/ppo-pilot-001`: completed run configuration,
  input hashes, frozen source, split positions, selected actor checkpoints,
  per-seed choices, learning curves, and metrics.
- `/var/tmp/mshanthappa/rl-cs/upstream`: separate baseline source checkouts.
- `/var/tmp/mshanthappa/rl-cs/data/ogbn-products.pt`: existing local graph
  asset; not opened by this pilot and not yet verified for baseline training.

Large artifacts live outside the nearly full home quota. `/var/tmp` storage
is machine-local and should not be treated as an archival release. Existing
Products and robotics projects are unchanged. Nothing has been published.
