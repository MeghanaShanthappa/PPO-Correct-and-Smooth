# Completed first GraDBERT/GIANT comparison

The four-stage author-source baseline, fixed C&S, and PPO comparison have
completed. This is one base-model seed (0), five controller seeds (20–24),
and a diagnostic subset of the historically validation-informed official
validation split. It is not an official test result or a multi-base-seed claim.

The final model's full-validation accuracy before C&S was 93.6373%; fixed C&S
scored 93.6856%. The policy comparison below uses a different denominator:
the same 7,865 diagnostic validation nodes for every method.

| Method | Diagnostic accuracy | Net correct versus fixed C&S |
|---|---:|---:|
| Raw GraDBERT/SAGN predictions | 94.2022% | +12 |
| Fixed C&S | 94.0496% | 0 |
| Supervised selector, five-seed mean | 94.0496% | 0 |
| PPO selector, five-seed mean | 94.0623% | +1 |

PPO's gains by actor seed were 0, 0, 1, 2 and 2 nodes. The mean gain is
0.0127 percentage points, and the standard deviation of PPO accuracy across
these seeds is also 0.0127 percentage points. This is variation in actor
training on a single fixed base, not independent base-model replications.

The 0.0127-point mean increase over fixed C&S is an exploratory point estimate.
PPO still trails simply using raw predictions on this subset. Therefore this
experiment does not establish a robust PPO advantage or an improvement over
the strongest available fixed action. An oracle choosing the correct action
per node scores about 94.4565%, but it uses evaluation labels and cannot be
deployed.

The selection split retained the initial always-C&S actor for two PPO seeds
and all supervised seeds. All seeds were reported; none was selected based on
the diagnostic holdout. Official test labels were masked in base training/C&S
and were absent from policy exports.

## Run records

- Base stages: `/var/tmp/mshanthappa/rl-cs/runs/grad-seed0-main/stages.json`
- C&S settings/provenance: `/var/tmp/mshanthappa/rl-cs/runs/grad-seed0-actions/manifest.json`
- Controller results: `/var/tmp/mshanthappa/rl-cs/runs/grad-seed0-ppo/summary.json`
- Per-seed outcomes: `/var/tmp/mshanthappa/rl-cs/runs/grad-seed0-ppo/runs.json`

The persistent launcher stopped after successful base training due to a shell
handoff error. C&S and policy training were resumed directly from the saved
stage-3 predictions. No baseline retraining or test scoring was needed. The
launcher now skips completed stages when rerun; the historical failure marker
is retained separately after successful recovery.

The generic pilot initially labeled this run with its old hybrid-stack scope
text. That description was corrected from the actual input manifests; original
reports and a metadata-amendment record are preserved. Input predictions,
policy choices and checkpoint selection were not changed.

## Subsequent tie-breaking audit

While preparing V2, six of the 31,458 allowed development rows were found to
have exact tied top C&S scores. V1's cached exporter took the first index
returned by `torch.topk`; V2 uses the lowest-index maximum consistently through
`argmax`. Every observed difference was an exact tie, rather than a changed
propagation result. Future exports now use `argmax`, with a regression test.
The original V1 metrics above are preserved under their original convention;
the previously viewed diagnostic cohort was not rescored. Do not attribute
differences between these conventions to PPO or compare V2 selection metrics
directly with V1 diagnostic accuracy.
