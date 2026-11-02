import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import dgl

from run_grad_baseline import masked_labels
from postprocess_grad import fix_correction_order, export_action
import numpy as np


class GradAdapterTests(unittest.TestCase):
    def test_prediction_export_uses_argmax_for_equal_class_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            export_action(Path(folder),'raw',torch.ones(1,47)/47,
                          torch.tensor([],dtype=torch.long),torch.tensor([0]),torch.tensor([0]),0)
            with np.load(Path(folder)/'raw.npz') as z:
                self.assertEqual(z['prediction'].tolist(),[0])

    def test_official_test_labels_are_masked_without_changing_training_targets(self):
        original = torch.tensor([2, 1, 0, 3])
        masked = masked_labels(original, torch.tensor([1, 3]))
        torch.testing.assert_close(masked, torch.tensor([2, -1, 0, -1]))
        torch.testing.assert_close(original, torch.tensor([2, 1, 0, 3]))

    def test_correction_respects_nonidentity_split_order(self):
        upstream = Path('/var/tmp/mshanthappa/rl-cs/upstream/GRAD/GradBERT/ogbn-products/SAGN_with_SLE/src')
        sys.path.insert(0,str(upstream))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'patched_models.py'
            path.write_text(fix_correction_order((upstream/'models.py').read_text()))
            spec = importlib.util.spec_from_file_location('patched_models_test',path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            graph = dgl.graph(([0, 1, 2], [0, 1, 2]))
            probabilities = torch.tensor([[.8, .2], [.1, .9], [.5, .5]])
            train, valid, test = [torch.tensor([i]) for i in [2, 0, 1]]
            cs = module.CorrectAndSmooth(1,0,'DAD',1,0,'DAD',autoscale=False,scale=1)
            actual = cs.correct(graph,probabilities,torch.tensor([1]),train,valid,test,3)
            torch.testing.assert_close(actual,torch.tensor([[0.,1.],[.1,.9],[.5,.5]]))


if __name__=='__main__':
    unittest.main()
