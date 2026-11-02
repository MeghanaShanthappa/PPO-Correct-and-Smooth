# Cross-dataset local C&S versus PPO

Validation is used for checkpoint selection; these scores are development results.

| Dataset | Base seeds | Fixed C&S | PPO C&S | Gain (percentage points) |
|---|---:|---:|---:|---:|
| ogbn-products | 3/3 | 91.518% | 91.842% | +0.324 |
| ogbn-arxiv | 3/3 | 70.426% | 70.748% | +0.321 |

Within each base seed, PPO accuracy averages five actor seeds. Base models are trained independently per dataset; this tests method transfer, not zero-shot policy transfer.

Products uses its existing spectral-augmented MLP. Arxiv uses standardized official 128-dimensional features without added spectral/diffusion features. This is not an arxiv leaderboard-baseline reproduction.

## Frozen arxiv final test

Fixed C&S: 68.804%. PPO C&S: 69.064%.
Paired mean gain: +0.260 percentage points across three base seeds.
Validation-best fixed action: 68.885%.
This completed result must be reported whether positive or negative. No settings change after test access.

The earlier, different Products PPO experiment lost 2.467 test points; that result remains part of the record. The new Products small-adjustment experiment has no test score.
