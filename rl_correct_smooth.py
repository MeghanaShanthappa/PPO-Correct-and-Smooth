#!/usr/bin/env python3
"""RL-controlled Correct & Smooth for transductive node classification.

This is deliberately a research prototype: graph propagation and the policy are
implemented with plain PyTorch rather than a prebuilt GNN or RL package.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ACTIONS = ("KEEP", "CORRECT", "SMOOTH", "CONTRAST", "STOP")
KEEP, CORRECT, SMOOTH, CONTRAST, STOP = range(len(ACTIONS))


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_idx: torch.Tensor
    valid_idx: torch.Tensor
    test_idx: torch.Tensor


@dataclass
class Config:
    seed: int = 7
    hidden: int = 64
    base_epochs: int = 250
    policy_epochs: int = 300
    policy_steps: int = 4
    lr: float = 0.01
    policy_lr: float = 0.003
    anchor_fraction: float = 0.75
    correct_strength: float = 1.0
    smooth_strength: float = 0.65
    contrast_strength: float = 0.35
    compute_cost: float = 0.002
    entropy_weight: float = 0.005


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    """Add reverse edges and remove duplicate directed edges."""
    both = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return torch.unique(both, dim=1)


class GraphOps:
    """Sparse mean aggregation using only PyTorch index operations."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        self.src, self.dst = edge_index.long()
        self.num_nodes = num_nodes
        degree = torch.bincount(self.dst, minlength=num_nodes).float()
        self.degree = degree.clamp_min(1.0)
        self.degree_feature = torch.log1p(degree) / torch.log1p(degree.max().clamp_min(1.0))

    def to(self, device: torch.device) -> "GraphOps":
        self.src = self.src.to(device)
        self.dst = self.dst.to(device)
        self.degree = self.degree.to(device)
        self.degree_feature = self.degree_feature.to(device)
        return self

    def mean(self, values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values)
        result.index_add_(0, self.dst, values[self.src])
        shape = (self.num_nodes,) + (1,) * (values.ndim - 1)
        return result / self.degree.view(shape)


class BaseMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActionEngine:
    """Construct the five possible next predictions for every node."""

    def __init__(
        self,
        graph: GraphOps,
        labels: torch.Tensor,
        anchor_idx: torch.Tensor,
        classes: int,
        cfg: Config,
    ) -> None:
        self.graph = graph
        self.labels = labels
        self.anchor_idx = anchor_idx
        self.classes = classes
        self.cfg = cfg
        self.anchor_targets = F.one_hot(labels[anchor_idx], classes).float()

    @staticmethod
    def normalize(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp_min(1e-8)
        return values / values.sum(dim=1, keepdim=True)

    def seeded(self, prediction: torch.Tensor) -> torch.Tensor:
        seeded = prediction.clone()
        seeded[self.anchor_idx] = self.anchor_targets
        return seeded

    def candidates(self, prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seeded = self.seeded(prediction)
        neighbor_prediction = self.graph.mean(seeded)

        residual = torch.zeros_like(prediction)
        residual[self.anchor_idx] = self.anchor_targets - prediction[self.anchor_idx]
        correction = self.graph.mean(residual)

        correct = self.normalize(prediction + self.cfg.correct_strength * correction)
        smooth = self.normalize(
            (1.0 - self.cfg.smooth_strength) * prediction
            + self.cfg.smooth_strength * neighbor_prediction
        )
        contrast = self.normalize(
            prediction + self.cfg.contrast_strength * (prediction - neighbor_prediction)
        )
        # KEEP and STOP have identical values. STOP additionally deactivates a node.
        options = torch.stack((prediction, correct, smooth, contrast, prediction), dim=1)
        correction_magnitude = (correct - prediction).abs().mean(dim=1, keepdim=True)
        return options, correction_magnitude


class NodePolicy(nn.Module):
    def __init__(self, classes: int, hidden: int = 64) -> None:
        super().__init__()
        state_size = 2 * classes + 6
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def policy_state(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    graph: GraphOps,
    correction_magnitude: torch.Tensor,
    step: int,
    max_steps: int,
) -> torch.Tensor:
    neighbor = graph.mean(prediction)
    confidence = prediction.max(dim=1, keepdim=True).values
    entropy = -(prediction * prediction.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    entropy = entropy / math.log(prediction.size(1))
    disagreement = (prediction - neighbor).abs().mean(dim=1, keepdim=True)
    degree = graph.degree_feature.unsqueeze(1)
    step_feature = torch.full_like(confidence, step / max(max_steps - 1, 1))
    return torch.cat(
        (prediction, base_prediction, confidence, entropy, disagreement,
         degree, correction_magnitude, step_feature), dim=1
    )


@dataclass
class Rollout:
    prediction: torch.Tensor
    node_log_probability: torch.Tensor
    mean_entropy: torch.Tensor
    used_steps: torch.Tensor
    action_counts: torch.Tensor


def rollout(
    policy: NodePolicy,
    engine: ActionEngine,
    base_prediction: torch.Tensor,
    max_steps: int,
    sample: bool,
) -> Rollout:
    prediction = base_prediction.clone()
    nodes = prediction.size(0)
    active = torch.ones(nodes, dtype=torch.bool, device=prediction.device)
    log_probability = torch.zeros(nodes, device=prediction.device)
    decision_count = torch.zeros(nodes, device=prediction.device)
    used_steps = torch.zeros(nodes, device=prediction.device)
    entropy_total = torch.zeros((), device=prediction.device)
    entropy_count = 0
    action_counts = torch.zeros(len(ACTIONS), dtype=torch.long, device=prediction.device)

    for step in range(max_steps):
        options, correction_magnitude = engine.candidates(prediction)
        state = policy_state(
            prediction, base_prediction, engine.graph, correction_magnitude, step, max_steps
        )
        distribution = torch.distributions.Categorical(logits=policy(state))
        action = distribution.sample() if sample else distribution.probs.argmax(dim=1)
        action = torch.where(active, action, torch.full_like(action, STOP))

        chosen = options[torch.arange(nodes, device=prediction.device), action]
        prediction = torch.where(active.unsqueeze(1), chosen, prediction)

        current_log_probability = distribution.log_prob(action)
        log_probability = log_probability + current_log_probability * active
        decision_count = decision_count + active.float()
        entropy_total = entropy_total + (distribution.entropy() * active).sum()
        entropy_count += int(active.sum())
        used_steps = used_steps + (active & (action != STOP)).float()
        action_counts += torch.bincount(action[active], minlength=len(ACTIONS))
        active = active & (action != STOP)
        if not bool(active.any()):
            break

    return Rollout(
        prediction=prediction,
        node_log_probability=log_probability / decision_count.clamp_min(1.0),
        mean_entropy=entropy_total / max(entropy_count, 1),
        used_steps=used_steps,
        action_counts=action_counts,
    )


def accuracy(prediction: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    return float((prediction[idx].argmax(1) == labels[idx]).float().mean())


def fixed_actions(
    engine: ActionEngine, base_prediction: torch.Tensor, actions: Tuple[int, ...]
) -> torch.Tensor:
    """Apply the same hand-written action schedule to every node."""
    prediction = base_prediction.clone()
    for action in actions:
        options, _ = engine.candidates(prediction)
        prediction = options[:, action]
    return prediction


def split_anchors(train_idx: torch.Tensor, fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(train_idx.numel(), generator=generator).to(train_idx.device)
    shuffled = train_idx[permutation]
    cut = min(max(int(len(shuffled) * fraction), 1), len(shuffled) - 1)
    return shuffled[:cut], shuffled[cut:]


def train_base(
    data: GraphData, train_idx: torch.Tensor, cfg: Config, classes: int
) -> Tuple[BaseMLP, torch.Tensor]:
    model = BaseMLP(data.x.size(1), cfg.hidden, classes).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    for _ in range(cfg.base_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(data.x).softmax(dim=1)
    return model, prediction


def train_policy(
    policy: NodePolicy,
    engine: ActionEngine,
    base_prediction: torch.Tensor,
    reward_idx: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
) -> None:
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    base_log_true = base_prediction[reward_idx, labels[reward_idx]].clamp_min(1e-8).log()
    moving_baseline = 0.0
    best_score = -float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    evaluation_interval = max(cfg.policy_epochs // 25, 1)

    for epoch in range(1, cfg.policy_epochs + 1):
        policy.train()
        result = rollout(policy, engine, base_prediction, cfg.policy_steps, sample=True)
        final_log_true = result.prediction[reward_idx, labels[reward_idx]].clamp_min(1e-8).log()
        reward = final_log_true - base_log_true
        reward = reward - cfg.compute_cost * result.used_steps[reward_idx]

        episode_reward = float(reward.mean().detach())
        moving_baseline = 0.95 * moving_baseline + 0.05 * episode_reward
        advantage = reward.detach() - moving_baseline
        local_loss = -(result.node_log_probability[reward_idx] * advantage).mean()
        # A small global term gives credit to actions on neighboring, unlabeled nodes.
        global_loss = -0.1 * result.node_log_probability.mean() * (episode_reward - moving_baseline)
        loss = local_loss + global_loss - cfg.entropy_weight * result.mean_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        if epoch == 1 or epoch % evaluation_interval == 0 or epoch == cfg.policy_epochs:
            policy.eval()
            with torch.no_grad():
                deterministic = rollout(
                    policy, engine, base_prediction, cfg.policy_steps, sample=False
                )
                deterministic_log_true = deterministic.prediction[
                    reward_idx, labels[reward_idx]
                ].clamp_min(1e-8).log()
                score = float(
                    (deterministic_log_true - base_log_true
                     - cfg.compute_cost * deterministic.used_steps[reward_idx]).mean()
                )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(policy.state_dict())

        if epoch == 1 or epoch % max(cfg.policy_epochs // 5, 1) == 0:
            improved = accuracy(result.prediction, labels, reward_idx)
            base_acc = accuracy(base_prediction, labels, reward_idx)
            print(
                f"policy epoch {epoch:4d} | reward {episode_reward:+.4f} | "
                f"masked accuracy {base_acc:.3f}->{improved:.3f}"
            )

    policy.load_state_dict(best_state)


def make_synthetic_graph(seed: int = 7) -> GraphData:
    """A mixed homophilic/heterophilic graph suitable for a fast smoke test."""
    generator = torch.Generator().manual_seed(seed)
    classes, per_class, features = 3, 40, 8
    labels = torch.arange(classes).repeat_interleave(per_class)
    centers = F.one_hot(torch.arange(classes), features).float() * 2.0
    x = centers[labels] + 0.9 * torch.randn(classes * per_class, features, generator=generator)

    edges = []
    for node in range(len(labels)):
        same = torch.where(labels == labels[node])[0]
        other = torch.where(labels != labels[node])[0]
        for neighbor in same[torch.randperm(len(same), generator=generator)[:4]]:
            if int(neighbor) != node:
                edges.append((node, int(neighbor)))
        # Every fifth node has deliberately heterophilous connections.
        count = 4 if node % 5 == 0 else 1
        for neighbor in other[torch.randperm(len(other), generator=generator)[:count]]:
            edges.append((node, int(neighbor)))
    edge_index = make_bidirectional(torch.tensor(edges).t().contiguous())

    train, valid, test = [], [], []
    for label in range(classes):
        idx = torch.where(labels == label)[0]
        idx = idx[torch.randperm(len(idx), generator=generator)]
        train.append(idx[:15]); valid.append(idx[15:25]); test.append(idx[25:])
    return GraphData(x, edge_index, labels, torch.cat(train), torch.cat(valid), torch.cat(test))


def load_ogb(name: str, root: str) -> GraphData:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise SystemExit("Install OGB dependencies: pip install ogb torch-geometric") from exc
    dataset = PygNodePropPredDataset(name=name, root=root)
    graph = dataset[0]
    split = dataset.get_idx_split()
    valid = split.get("valid", split.get("val"))
    return GraphData(
        graph.x.float(), make_bidirectional(graph.edge_index), graph.y.view(-1).long(),
        split["train"], valid, split["test"],
    )


def move_data(data: GraphData, device: torch.device) -> GraphData:
    return GraphData(*(value.to(device) for value in asdict(data).values()))


def run(data: GraphData, cfg: Config, output: Path) -> Dict[str, object]:
    classes = int(data.y.max()) + 1
    anchor_idx, reward_idx = split_anchors(data.train_idx, cfg.anchor_fraction, cfg.seed)
    _, base_prediction = train_base(data, anchor_idx, cfg, classes)
    graph = GraphOps(data.edge_index, data.x.size(0)).to(data.x.device)
    engine = ActionEngine(graph, data.y, anchor_idx, classes, cfg)
    policy = NodePolicy(classes, cfg.hidden).to(data.x.device)
    train_policy(policy, engine, base_prediction, reward_idx, data.y, cfg)

    policy.eval()
    with torch.no_grad():
        final = rollout(policy, engine, base_prediction, cfg.policy_steps, sample=False)
        correct_only = fixed_actions(engine, base_prediction, (CORRECT,) * cfg.policy_steps)
        smooth_only = fixed_actions(engine, base_prediction, (SMOOTH,) * cfg.policy_steps)
        correct_steps = max(cfg.policy_steps // 2, 1)
        smooth_steps = max(cfg.policy_steps - correct_steps, 1)
        fixed_cs = fixed_actions(
            engine, base_prediction,
            (CORRECT,) * correct_steps + (SMOOTH,) * smooth_steps,
        )

    results: Dict[str, object] = {
        "base_valid_accuracy": accuracy(base_prediction, data.y, data.valid_idx),
        "fixed_correct_valid_accuracy": accuracy(correct_only, data.y, data.valid_idx),
        "fixed_smooth_valid_accuracy": accuracy(smooth_only, data.y, data.valid_idx),
        "fixed_cs_valid_accuracy": accuracy(fixed_cs, data.y, data.valid_idx),
        "policy_valid_accuracy": accuracy(final.prediction, data.y, data.valid_idx),
        "base_test_accuracy": accuracy(base_prediction, data.y, data.test_idx),
        "fixed_correct_test_accuracy": accuracy(correct_only, data.y, data.test_idx),
        "fixed_smooth_test_accuracy": accuracy(smooth_only, data.y, data.test_idx),
        "fixed_cs_test_accuracy": accuracy(fixed_cs, data.y, data.test_idx),
        "policy_test_accuracy": accuracy(final.prediction, data.y, data.test_idx),
        "mean_steps": float(final.used_steps.mean()),
        "action_counts": {name: int(count) for name, count in zip(ACTIONS, final.action_counts)},
        "config": asdict(cfg),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "config": asdict(cfg)}, output / "policy.pt")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic", choices=("synthetic", "ogbn-arxiv"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--base-epochs", type=int, default=Config.base_epochs)
    parser.add_argument("--policy-epochs", type=int, default=Config.policy_epochs)
    parser.add_argument("--policy-steps", type=int, default=Config.policy_steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed, base_epochs=args.base_epochs,
        policy_epochs=args.policy_epochs, policy_steps=args.policy_steps,
    )
    seed_everything(cfg.seed)
    if args.device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = args.device
    device = torch.device(selected)
    data = make_synthetic_graph(cfg.seed) if args.dataset == "synthetic" else load_ogb(args.dataset, args.data_root)
    data = move_data(data, device)
    print(f"dataset={args.dataset} nodes={len(data.y)} edges={data.edge_index.size(1)} device={device}")
    run(data, cfg, Path(args.output))


if __name__ == "__main__":
    main()
