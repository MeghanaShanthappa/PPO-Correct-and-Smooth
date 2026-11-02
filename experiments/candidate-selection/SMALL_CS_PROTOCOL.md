# Experiment 1: modest adjustments around original C&S

Run only this first idea before introducing a different reward or training
anchor randomization. The prior frozen test failure informed the follow-up;
these are exploratory validation results, not an untouched confirmation.

Reuse frozen base MLP seeds 0, 1, 2 and their original three-fold OOF MLPs.
Keep original official train/validation/test memberships. PPO rewards use
OOF official training targets only. All C&S anchors use official training
labels, excluding the target fold. Validation is used only for checkpoint
and hyperparameter selection. No test labels or new test scoring are needed.

Five actions: correction scale 9, original scale 10, scale 11 (all smoothing
alpha .8), and smoothing alpha .75 or .85 (both scale 10). All retain original
50/50 steps, correction alpha 1, DAD/DA normalization. Raw MLP is not an action.
These are modest parameter changes, not a guaranteed bound on prediction
differences. Baseline occupies index 1 as in the existing training code.

State uses the generic per-action statistics, genuine raw-MLP predicted class,
and the existing five structural statistics. Its width is 87 for five actions.
The actor remains hidden-width 32; all learning methods share its architecture.
Reward, training cap, five actor seeds, warm start, and optimization settings
remain those of the original experiment. The changed action bank is the
experimental variable; state width necessarily changes with action count.

Compare fixed C&S, validation-best global candidate, warm-start, supervised,
expected-reward and PPO controllers. Verify recomputed fixed-C&S probabilities
against the cached development bank before training any policy. Preserve the
original locked source and results. Report paired differences across the
three base seeds; five actor runs are not five independent base models.

Run data: `/var/tmp/mshanthappa/rl-cs/small-cs-v1`.
Log: `small-cs.log`. Summary: the run directory's `summary.json`.
Do not automatically start another idea or evaluate official test labels.
