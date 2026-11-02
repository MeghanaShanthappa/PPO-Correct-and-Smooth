import unittest
from types import SimpleNamespace
import numpy as np
import torch
from mlp_support import original_mlp,original_cs,spectral_dense,anchor_labels


class DenseAdj:
    def __init__(self,a): self.a=a
    def to(self,device): return self
    def __matmul__(self,x): return self.a@x


class MLPReproductionTests(unittest.TestCase):
    def test_masked_labels_keep_public_47_class_output_width(self):
        labels=torch.tensor([[0],[-1],[-1]])
        probabilities=torch.ones(3,47)/47
        residual=original_cs().pre_residual_correlation(labels,probabilities,torch.tensor([0]))
        self.assertEqual(tuple(residual.shape),(3,47))

    def test_heldout_targets_cannot_enter_correction_anchors(self):
        anchors=torch.tensor([0,2])
        torch.testing.assert_close(anchor_labels(torch.tensor([0,1,2,3]),anchors),
                                   anchor_labels(torch.tensor([0,9,2,8]),anchors))

    def test_original_model_matches_leaderboard_parameter_count(self):
        model=original_mlp()(228,200,47,3,.5,True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),96247)

    def test_spectral_operator_preserves_stationary_eigenvector(self):
        a=np.array([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
        op=spectral_dense(a)
        degree=a.sum(1); tau=degree.mean()
        v=np.sqrt(degree+tau)
        np.testing.assert_allclose(op@v,2*v,atol=1e-12)
        np.testing.assert_allclose(op,op.T)

    def test_original_cs_matches_explicit_dense_iterations(self):
        a=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
        degree=a.sum(1)
        dad=degree.rsqrt()[:,None]*a*degree.rsqrt()[None,:]
        da=a/degree[:,None]
        labels=torch.tensor([[0],[1],[-1]])
        p=torch.tensor([[.7,.3],[.2,.8],[.6,.4]])
        split={'train':torch.tensor([0,1]),'valid':torch.tensor([2]),'test':torch.tensor([],dtype=torch.long)}
        c,r=original_cs().double_correlation_fixed(SimpleNamespace(y=labels),p,split,
            DenseAdj(dad),1.,3,DenseAdj(da),.8,3,scale=10,train_only=True,device='cpu',display=False)
        error=torch.zeros_like(p); error[:2]=torch.eye(2)-p[:2]
        v=error.clone()
        for _ in range(3): v=dad@v; v[:2]=error[:2]
        expected=p+10*v
        torch.testing.assert_close(c,expected)
        y=expected.clone(); y[:2]=torch.eye(2); v=y.clone()
        for _ in range(3): v=(.8*(da@v)+.2*y).clamp(0,1)
        torch.testing.assert_close(r,v)


if __name__=='__main__': unittest.main()
