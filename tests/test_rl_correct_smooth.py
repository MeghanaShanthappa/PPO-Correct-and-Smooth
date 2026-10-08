import unittest

import torch

from rl_correct_smooth import (
    ACTIONS,
    ActionEngine,
    Config,
    GraphOps,
    NodePolicy,
    make_synthetic_graph,
    rollout,
)


class RLCorrectSmoothTests(unittest.TestCase):
    def setUp(self):
        self.data = make_synthetic_graph(3)
        self.graph = GraphOps(self.data.edge_index, len(self.data.y))
        self.cfg = Config(policy_steps=3)
        self.classes = int(self.data.y.max()) + 1
        self.base = torch.full((len(self.data.y), self.classes), 1 / self.classes)
        self.engine = ActionEngine(
            self.graph, self.data.y, self.data.train_idx[:20], self.classes, self.cfg
        )

    def test_candidates_are_probabilities(self):
        candidates, magnitude = self.engine.candidates(self.base)
        self.assertEqual(candidates.shape, (len(self.data.y), len(ACTIONS), self.classes))
        self.assertTrue(torch.allclose(candidates.sum(-1), torch.ones_like(candidates.sum(-1))))
        self.assertEqual(magnitude.shape, (len(self.data.y), 1))

    def test_rollout_is_finite(self):
        policy = NodePolicy(self.classes, hidden=16)
        result = rollout(policy, self.engine, self.base, max_steps=3, sample=False)
        self.assertTrue(torch.isfinite(result.prediction).all())
        self.assertTrue(torch.allclose(result.prediction.sum(1), torch.ones(len(self.data.y))))
        decisions = int(result.action_counts.sum())
        self.assertGreaterEqual(decisions, len(self.data.y))
        self.assertLessEqual(decisions, len(self.data.y) * 3)


if __name__ == "__main__":
    unittest.main()
