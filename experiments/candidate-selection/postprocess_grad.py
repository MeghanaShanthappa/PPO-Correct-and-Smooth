"""Export validation-only actions from a completed GraDBERT/SAGN base run.

Uses the released active one-layer C&S settings and explicitly fixes an
upstream node-order bug in the non-autoscale correction path.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import torch


def fix_correction_order(source):
    old = 'result = y_all + self.scale * smoothed_error[torch.cat([train_nid, val_nid, test_nid], dim=0)]'
    new = 'result = y_soft + self.scale * smoothed_error[torch.cat([train_nid, val_nid, test_nid], dim=0)]'
    if source.count(old) != 1:
        raise ValueError('unrecognized upstream correction implementation')
    return source.replace(old, new)


def export_action(out, name, probabilities, train, valid, labels, seed):
    n = len(train) + len(valid)
    p = probabilities[:n].detach().cpu().float().clamp_min(0)
    p /= p.sum(1, keepdim=True).clamp_min(1e-12)
    top = p.topk(2, dim=1)
    prediction = p.argmax(1)
    np.savez_compressed(out / f'{name}.npz', prediction=prediction.numpy().astype(np.int16),
        confidence=top.values[:, 0].numpy().astype(np.float32),
        margin=(top.values[:, 0]-top.values[:, 1]).numpy().astype(np.float32),
        entropy=(-(p*p.clamp_min(1e-12).log()).sum(1)).numpy().astype(np.float32))
    accuracy = (prediction[len(train):] == labels[valid].cpu()).float().mean().item()
    metadata = {'action': name, 'seeds': [seed], 'train_rows': len(train), 'validation_rows': len(valid),
                'test_rows_exported': 0, 'validation_accuracy': accuracy,
                'feature_stack': 'authors GIANT stages 0-2 / GraDBERT-KD stage 3; no LD or EmbeddingGemma',
                'parameters': {} if name == 'raw' else {'correction_layers':1,'smoothing_layers':1,
                    'correction_alpha':.45,'smoothing_alpha':.6,'scale':20,'adjacency':'DAD','autoscale':False}}
    (out / f'{name}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root, out = Path(args.baseline_run), Path(args.output)
    if not (root/'BASE_TRAINING_COMPLETE').exists():
        raise RuntimeError('base training has not completed')
    out.mkdir(parents=True, exist_ok=False)
    source = (root/'source/models.py').read_text()
    fixed = fix_correction_order(source)
    (out/'models_with_order_fix.py').write_text(fixed)
    config = json.loads((root/'config.json').read_text())
    sys.path.insert(0,str(root/'source'))
    spec = importlib.util.spec_from_file_location('grad_cs_models',out/'models_with_order_fix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = DglNodePropPredDataset(name='ogbn-products',root=args.data_root)
    graph, labels = data[0]
    split = data.get_idx_split()
    train, valid, test = (split[k] for k in ['train','valid','test'])
    labels = labels.squeeze()
    labels[test] = -1
    graph.ndata.clear()
    probs_path = root/f"seed-{config['seed']}-stage-3.pt"
    probabilities = torch.load(probs_path,map_location='cpu')
    assert probabilities.shape == (2449029,47)
    np.save(out/'train_valid_labels.npy',labels[torch.cat([train,valid])].numpy(),allow_pickle=False)
    records = [export_action(out,'raw',probabilities,train,valid,labels,config['seed'])]
    cs = module.CorrectAndSmooth(1,.45,'DAD',1,.6,'DAD',autoscale=False,scale=20)
    with torch.no_grad():
        processed = cs(graph,probabilities,labels[train],['correction','smoothing'],train,valid,test,len(labels))
    records.append(export_action(out,'fixed_cs',processed,train,valid,labels,config['seed']))
    manifest = {'baseline_run':str(root.resolve()),'test_labels_masked_before_processing':True,'test_scored':False,
                'correction_node_order_fix':True,'active_released_script_settings':True,
                'published_score_reproduction_not_guaranteed':True,
                'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'models_original_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'models_adapted_sha256':hashlib.sha256(fixed.encode()).hexdigest(),
                'actions':records}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
