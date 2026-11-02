"""Experiment 1: local C&S parameter changes, cached frozen MLPs, train OOF only."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
import torch

from mlp_support import GPUAdjacency, anchor_labels, original_cs
from prepare_v2 import state_from_probabilities

BASE = Path('/var/tmp/mshanthappa/rl-cs/mlp')
ACTIONS = [('scale9', (9., .8)), ('fixed_cs', (10., .8)),
           ('scale11', (11., .8)), ('smooth075', (10., .75)),
           ('smooth085', (10., .85))]


def policy_state(bank, raw, structural):
    state, names = state_from_probabilities(bank, structural, [n for n, _ in ACTIONS])
    # Generic constructor assumes action 0 is raw. This bank has no raw action.
    cols = [i for i, name in enumerate(names) if name.startswith('raw_predicted_class_')]
    state[:, cols] = np.eye(47, dtype=np.float32)[raw.argmax(1)]
    return state, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    out = Path(args.output)
    if (out / 'manifest.json').exists():
        print('Prepared data already complete', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    source = BASE / f'seed-{args.seed}'
    with np.load(source / 'development.npz') as z:
        ids, labels, train, valid, oldstate = (z[k] for k in ['original_node_ids', 'labels', 'train', 'selection', 'state'])
        old = z['probabilities']
        raw_rows, reference = old[0].copy(), old[1].copy()
    assert len(train) == 196615 and len(valid) == 39323
    assert len(np.unique(ids)) == len(ids)
    with np.load(BASE / 'spectral/split-ids.npz') as z:
        assert np.array_equal(ids[train], z['train']) and np.array_equal(ids[valid], z['valid'])
        assert not np.intersect1d(ids, z['test']).size
    adj = sparse.load_npz(BASE / 'spectral/binary-adjacency.npz').astype(np.float32)
    degree = np.asarray(adj.sum(1)).ravel()
    inv = np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)
    gpu = csr_matrix(adj)
    d = cp.asarray(np.sqrt(inv))
    dad = GPUAdjacency(gpu.multiply(d[:, None]).multiply(d[None, :]).tocsr())
    da = GPUAdjacency(gpu.multiply(cp.asarray(inv)[:, None]).tocsr())
    # Construct targets only from the already isolated train/validation package.
    y = torch.full((adj.shape[0],), -1, dtype=torch.long)
    y[torch.from_numpy(ids[train])] = torch.from_numpy(labels[train])
    bank = np.empty((len(ACTIONS), len(ids), 47), dtype=np.float32)
    cs = original_cs()

    def fill(model_file, anchors, positions):
        raw = torch.load(model_file, map_location='cpu')
        targets = torch.from_numpy(ids[positions])
        masked = anchor_labels(y, anchors)
        assert not np.intersect1d(anchors.numpy(), targets.numpy()).size
        split = {'train': anchors, 'valid': torch.empty(0, dtype=torch.long), 'test': torch.empty(0, dtype=torch.long)}
        for a, (name, (scale, alpha)) in enumerate(ACTIONS):
            _, value = cs.double_correlation_fixed(SimpleNamespace(y=masked[:, None]), raw, split,
                dad, 1., 50, da, alpha, 50, scale=scale, train_only=True, device='cuda', display=False)
            selected = value[targets].clamp_min(0)
            selected /= selected.sum(1, keepdim=True).clamp_min(1e-12)
            bank[a, positions] = selected.numpy()
            print(f'Seed {args.seed} {model_file.stem}: {name}, {len(positions)} rows', flush=True)
        del raw
        torch.cuda.empty_cache()

    fill(source / 'full.pt', torch.from_numpy(ids[train]), valid)
    for fold in range(3):
        target_ids = np.load(source / f'fold-{fold}-target-node-ids.npy')
        lookup = {int(node): pos for pos, node in enumerate(ids[train])}
        positions = np.array([lookup[int(node)] for node in target_ids])
        anchors = torch.from_numpy(np.setdiff1d(ids[train], target_ids))
        fill(source / f'fold-{fold}.pt', anchors, positions)
    # Numerical reproduction gate before any new policy optimization.
    max_difference = float(np.max(np.abs(bank[1] - reference)))
    agreement = bool(np.array_equal(bank[1].argmax(1), reference.argmax(1)))
    assert max_difference < 1e-5 and agreement, (max_difference, agreement)
    state, names = policy_state(bank, raw_rows, oldstate[:, -5:])
    np.savez_compressed(out / 'development.npz', state=state, probabilities=bank, labels=labels,
                        train=train, selection=valid, original_node_ids=ids)
    accuracies = {name: float((bank[a, valid].argmax(1) == labels[valid]).mean()) for a, (name, _) in enumerate(ACTIONS)}
    manifest = {'protocol': 'mlp_train_oof_v1', 'experiment': 'small_cs_v1',
        'base_seed': args.seed, 'actions': ACTIONS, 'feature_names': names,
        'train_rows': len(train), 'selection_rows': len(valid), 'excluded_diagnostic_rows': 0,
        'official_test_scored': False, 'baseline_validation': accuracies,
        'fixed_cs_max_difference': max_difference, 'fixed_cs_argmax_identical': agreement,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_test_exposure': 'Earlier frozen PPO test failure informed this exploratory follow-up; no new test evaluation.'}
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['baseline_validation']), flush=True)


if __name__ == '__main__':
    main()
