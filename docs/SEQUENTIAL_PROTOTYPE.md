# RL Correct & Smooth

Research prototype for node-wise reinforcement-learning control of graph
post-processing. It does **not** use a built-in GNN or RL implementation.

At every step, each active node chooses one action:

- `KEEP`: retain the current prediction and reconsider next step.
- `CORRECT`: propagate residual errors from labeled anchor nodes.
- `SMOOTH`: move toward neighboring predictions.
- `CONTRAST`: move away from neighboring predictions for heterophily.
- `STOP`: retain the prediction and stop processing this node.

The policy observes the current and base predictions, confidence, entropy,
neighbor disagreement, normalized degree, correction magnitude, and step. It is
trained with REINFORCE on a masked subset of training labels. The reward is the
change in true-class log probability minus a small computation cost.

## Quick smoke test

```bash
python3 -m unittest discover -s tests -v
python3 rl_correct_smooth.py \
  --dataset synthetic \
  --base-epochs 200 \
  --policy-epochs 300 \
  --output outputs/synthetic
```

## OGBN-Arxiv

Install dependencies in your Python environment:

```bash
python3 -m pip install -r requirements.txt
```

Run the experiment:

```bash
python3 rl_correct_smooth.py \
  --dataset ogbn-arxiv \
  --data-root "$HOME/ogb-data" \
  --base-epochs 500 \
  --policy-epochs 500 \
  --policy-steps 5 \
  --output outputs/ogbn-arxiv
```

The command writes `metrics.json` and `policy.pt`. Metrics include the base MLP,
fixed correction, fixed smoothing, fixed Correct-and-Smooth, and learned policy.
The OGB data directory can
be large, so keep it on persistent high-capacity storage rather than `$HOME` or
a temporary filesystem.

## Important experimental note

The base MLP and correction anchors use only part of the official training
labels. The remaining training labels are hidden and used as policy rewards.
Validation and test labels are never used to train the policy. This first
prototype should be compared against fixed C&S, NDLS, and a differentiable
action gate before making performance or novelty claims.

