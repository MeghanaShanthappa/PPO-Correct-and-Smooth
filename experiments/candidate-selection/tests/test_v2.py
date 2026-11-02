import unittest
import numpy as np
import torch
from prepare_v2 import state_from_probabilities
from train_v2 import training_reward,supervised_targets


class V2Tests(unittest.TestCase):
    def test_state_accepts_no_labels_and_does_not_contain_correctness(self):
        rng=np.random.default_rng(1)
        p=rng.random((6,8,47)).astype(np.float32)
        p/=p.sum(2,keepdims=True)
        state,names=state_from_probabilities(p,np.zeros((8,5)))
        self.assertEqual(state.shape,(8,len(names)))
        self.assertTrue(np.isfinite(state).all())
        self.assertFalse(any('true_class' in n or 'correctness' in n for n in names))

    def test_dense_reward_distinguishes_same_argmax_actions(self):
        p=np.array([[[.7,.3]],[[.9,.1]]],dtype=np.float32)
        r=training_reward(p,np.array([0]))
        self.assertGreater(r[0,1],r[0,0])

    def test_tied_best_actions_share_supervision(self):
        r=torch.tensor([[1.,1.,0.],[0.,0.,0.]])
        torch.testing.assert_close(supervised_targets(r),torch.tensor([[.5,.5,0.],[1/3,1/3,1/3]]))


if __name__=='__main__': unittest.main()
