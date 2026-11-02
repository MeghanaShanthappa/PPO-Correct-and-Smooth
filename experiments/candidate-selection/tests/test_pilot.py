import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from train_pilot import clipped_objective, split_rows, heldout_metrics, train, load_validation, ACTIONS


class ProtocolTests(unittest.TestCase):
    def test_every_row_has_exactly_one_role(self):
        a, b, c = split_rows(1000, 42)
        self.assertEqual([len(a), len(b), len(c)], [600, 200, 200])
        self.assertEqual(len(set(a) | set(b) | set(c)), 1000)
        np.testing.assert_array_equal(a, split_rows(1000, 42)[0])

    def test_ppo_clips_both_advantage_directions(self):
        actual = clipped_objective(torch.log(torch.tensor([2., .5])), torch.zeros(2), torch.tensor([1., -1.]), .2)
        torch.testing.assert_close(actual, torch.tensor([1.2, -.8]))

    def test_harm_and_help_count_separately(self):
        rewards = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        result = heldout_metrics(rewards, np.array([0, 0, 1, 1]))
        self.assertEqual(result['fixed_errors'], 1)
        self.assertEqual(result['introduced_errors'], 1)
        self.assertEqual(result['gain_pp'], 0)

    def test_test_containing_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ACTIONS:
                (Path(folder) / f'{name}.json').write_text(json.dumps({
                    'train_rows': 196615, 'validation_rows': 39323,
                    'test_rows_exported': 1, 'seeds': [10]}))
            with self.assertRaisesRegex(ValueError, 'test-containing'):
                load_validation(folder)

    def test_heldout_labels_cannot_change_training_or_checkpoint_selection(self):
        torch.set_num_threads(1)
        x = torch.arange(40, dtype=torch.float32).reshape(20, 2) / 40
        rewards = np.tile(np.array([[1., 0.], [0., 1.]], dtype=np.float32), (10, 1))
        changed = rewards.copy()
        changed[16:] = 1 - changed[16:]
        predictions = np.tile([0, 1], (20, 1))
        args = SimpleNamespace(hidden=4, lr=.01, epochs=3, update_epochs=2,
                               batch_size=4, clip=.2, entropy=.01)
        for method in ('supervised', 'ppo'):
            a, ma = train(method, 4, x, rewards, predictions, np.arange(12), np.arange(12, 16), args)
            b, mb = train(method, 4, x, changed, predictions, np.arange(12), np.arange(12, 16), args)
            self.assertEqual(ma['selected_epoch'], mb['selected_epoch'])
            for key in a.state_dict():
                torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
