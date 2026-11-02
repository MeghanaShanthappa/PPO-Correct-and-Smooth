import unittest
import numpy as np
from cross_dataset import state_features
from prepare_small_cs import policy_state
from train_v2 import training_reward, Actor
import torch


class CrossDatasetTests(unittest.TestCase):
    def make(self, classes):
        rng = np.random.default_rng(123)
        p = rng.random((5, 9, classes)).astype(np.float32)
        p /= p.sum(2, keepdims=True)
        raw = rng.random((9, classes)).astype(np.float32)
        raw /= raw.sum(1, keepdims=True)
        return p, raw, rng.random((9, 5)).astype(np.float32)

    def test_same_state_as_products_when_47_classes(self):
        args = self.make(47)
        state, names = state_features(*args)
        reference, refnames = policy_state(*args)
        np.testing.assert_array_equal(state, reference)
        self.assertEqual(names, refnames)

    def test_40_class_actor_reward_and_true_raw_encoding(self):
        p, raw, structural = self.make(40)
        state, names = state_features(p, raw, structural)
        self.assertEqual(state.shape, (9, 80))
        cols = [i for i, n in enumerate(names) if n.startswith('raw_predicted_class_')]
        np.testing.assert_array_equal(state[:, cols].argmax(1), raw.argmax(1))
        self.assertEqual(Actor(80, 32, 5)(torch.from_numpy(state)).shape, (9, 5))
        self.assertEqual(training_reward(p, np.arange(9)).shape, (9, 5))


if __name__ == '__main__':
    unittest.main()
