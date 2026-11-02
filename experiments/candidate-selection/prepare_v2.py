"""Build train/selection-only action bank and label-free graph state for PPO V2."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import dgl.function as fn
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import torch

from postprocess_grad import fix_correction_order
from train_pilot import split_rows

ACTIONS = [
    ('raw', 0, 0., 0., 0, 0.),
    ('fixed_cs', 1, .45, 20., 1, .6),
    ('weak_cs', 1, .2, 10., 1, .3),
    ('strong_cs', 1, .45, 20., 3, .8),
    ('smooth_only', 0, 0., 0., 1, .6),
    ('correct_only', 1, .45, 20., 0, 0.),
]


def normalized(p):
    p = p.float().clamp_min(0)
    return p / p.sum(1, keepdim=True).clamp_min(1e-12)


def state_from_probabilities(probabilities, structural, action_names=None):
    """No labels are accepted by the state constructor."""
    columns, names = [], []
    action_names=action_names or [a[0] for a in ACTIONS]
    features = []
    for a, p in enumerate(probabilities):
        top = np.sort(p, axis=1)[:, -2:]
        features.append(np.stack([top[:, 1], top[:, 1]-top[:, 0],
                                 -(p*np.log(np.maximum(p,1e-12))).sum(1)],1))
    for a, f in enumerate(features):
        columns.append(f)
        names.extend(f'{action_names[a]}_{n}' for n in ['confidence','margin','entropy'])
        columns.append(f-features[1])
        names.extend(f'{action_names[a]}_minus_fixed_{n}' for n in ['confidence','margin','entropy'])
        columns.append((probabilities[a].argmax(1)!=probabilities[1].argmax(1))[:,None])
        names.append(f'{action_names[a]}_disagrees_fixed')
    columns.append(np.eye(47,dtype=np.float32)[probabilities[0].argmax(1)])
    names.extend(f'raw_predicted_class_{i}' for i in range(47))
    columns.append(structural)
    names.extend(['log_degree','neighbor_support_raw','neighbor_class_agrees_raw',
                  'neighbor_entropy','raw_neighbor_probability_overlap'])
    state = np.concatenate(columns,axis=1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('nonfinite policy state')
    return state, names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    root,out=Path(args.baseline_run),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'config.json').read_text())
    sys.path.insert(0,str(root/'source'))
    source=(root/'source/models.py').read_text()
    (out/'models.py').write_text(fix_correction_order(source))
    spec=importlib.util.spec_from_file_location('v2_models',out/'models.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data=DglNodePropPredDataset(name='ogbn-products',root=args.data_root)
    graph,labels=data[0]; labels=labels.squeeze()
    split=data.get_idx_split()
    train,valid,test=(split[k] for k in ['train','valid','test'])
    labels[test]=-1
    policy,selection,excluded=split_rows(len(valid),20261028)
    allowed=np.concatenate([policy,selection])
    target_ids=valid[torch.from_numpy(allowed)]
    # OGB loads target arrays, but remove excluded targets before any scoring.
    labels[valid[torch.from_numpy(excluded)]]=-1
    graph.ndata.clear()
    path=root/f"seed-{config['seed']}-stage-3.pt"
    raw=torch.load(path,map_location='cpu')
    order=torch.cat([train,valid,test])
    positions=len(train)+torch.from_numpy(allowed)
    action_probs=[]
    for name,csteps,ca,scale,ssteps,sa in ACTIONS:
        if name=='raw':
            value=raw
        else:
            cs=module.CorrectAndSmooth(csteps,ca,'DAD',ssteps,sa,'DAD',autoscale=False,scale=scale)
            operations=(['correction'] if csteps else [])+(['smoothing'] if ssteps else [])
            value=cs(graph,raw,labels[train],operations,train,valid,test,len(labels))
        action_probs.append(normalized(value[positions]).numpy())
        print(f'Prepared {name} for {len(allowed)} allowed rows',flush=True)
        if name!='raw': del value
    raw_global=torch.empty_like(raw)
    raw_global[order]=normalized(raw)
    graph.ndata['p']=raw_global
    graph.update_all(fn.copy_u('p','m'),fn.mean('m','neighbor'))
    neighbor=normalized(graph.ndata.pop('neighbor')[target_ids])
    own=raw_global[target_ids]
    structural=torch.stack([graph.in_degrees(target_ids).float().log1p(),
        neighbor.gather(1,own.argmax(1,keepdim=True)).squeeze(1),
        (neighbor.argmax(1)==own.argmax(1)).float(),
        -(neighbor*neighbor.clamp_min(1e-12).log()).sum(1),(neighbor*own).sum(1)],1).numpy()
    probs=np.stack(action_probs)
    state,names=state_from_probabilities(probs,structural)
    np.savez_compressed(out/'development.npz',state=state,probabilities=probs,
        labels=labels[target_ids].numpy(),train=np.arange(len(policy)),
        selection=np.arange(len(policy),len(allowed)),original_validation_positions=allowed)
    manifest={**vars(args),'actions':ACTIONS,'feature_names':names,'base_seed':config['seed'],
        'train_rows':len(policy),'selection_rows':len(selection),'excluded_diagnostic_rows':len(excluded),
        'diagnostic_labels_scored':False,'official_test_scored':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_probabilities_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256((out/'development.npz').read_bytes()).hexdigest(),
        'scope':'development selection only; not an independent accuracy estimate'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'action_count':len(ACTIONS),'state_width':state.shape[1],
                     'training_rows':len(policy),'selection_rows':len(selection)}),flush=True)


if __name__=='__main__': main()
