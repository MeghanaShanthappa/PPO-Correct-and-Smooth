"""Load original model/C&S function definitions without unrelated PyG imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

UPSTREAM=Path('/var/tmp/mshanthappa/rl-cs/upstream/CorrectAndSmooth')


def load_definitions(filename,names):
    tree=ast.parse((UPSTREAM/filename).read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in selected}!=set(names): raise ValueError('upstream definition changed')
    if filename=='outcome_correlation.py':
        # Original code infers class count from ALL target labels. Masking
        # non-anchor targets must retain the public classifier output width.
        for node in selected:
            if node.name in ['pre_residual_correlation','pre_outcome_correlation']:
                for statement in node.body:
                    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='c' for t in statement.targets):
                        statement.value=ast.parse('model_out.shape[-1]',mode='eval').body
        ast.fix_missing_locations(tree)
        for node in selected: ast.fix_missing_locations(node)
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'tqdm':tqdm}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(UPSTREAM/filename),'exec'),scope)
    return SimpleNamespace(**{name:scope[name] for name in names})


def original_mlp(): return load_definitions('gen_models.py',['MLP']).MLP


def original_cs():
    return load_definitions('outcome_correlation.py',[
        'pre_residual_correlation','pre_outcome_correlation','general_outcome_correlation',
        'double_correlation_fixed'])


def anchor_labels(labels,anchors):
    masked=torch.full_like(labels,-1)
    masked[anchors]=labels[anchors]
    return masked


def spectral_dense(adjacency):
    degree=np.asarray(adjacency.sum(1)).reshape(-1)
    tau=float(degree.mean())
    d=1/np.sqrt(degree+tau)
    return np.eye(len(d))+d[:,None]*(adjacency+tau/len(d))*d[None,:]


class GPUAdjacency:
    """Original C&S adj.to()/adj@tensor interface backed by GPU CSR."""
    def __init__(self,matrix): self.matrix=matrix
    def to(self,device):
        if str(device) not in ['cuda','cuda:0']: raise ValueError('GPU adjacency requires cuda:0')
        return self
    def __matmul__(self,x):
        import cupy as cp
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            view=cp.fromDlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
            result=self.matrix@view
            return torch.utils.dlpack.from_dlpack(result.toDlpack())
