"""Frozen local-C&S protocol on ogbn-arxiv; prepare, fit, and locked final scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import torch.nn.functional as F

from mlp_support import GPUAdjacency, anchor_labels, original_cs, original_mlp
from prepare_small_cs import ACTIONS
from train_v2 import Actor

ROOT = Path('/var/tmp/mshanthappa/rl-cs/cross-dataset-v1/ogbn-arxiv')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def state_features(bank, raw, structural):
    """The same controller features, with a dataset-sized raw class encoding."""
    features = []
    names = []
    stats = []
    for p in bank:
        top = np.sort(p, axis=1)[:, -2:]
        stats.append(np.stack([top[:, 1], top[:, 1] - top[:, 0],
                               -(p * np.log(np.maximum(p, 1e-12))).sum(1)], 1))
    for a, (name, _) in enumerate(ACTIONS):
        features.extend([stats[a], stats[a] - stats[1],
                         (bank[a].argmax(1) != bank[1].argmax(1))[:, None]])
        names.extend(f'{name}_{n}' for n in ['confidence', 'margin', 'entropy'])
        names.extend(f'{name}_minus_fixed_{n}' for n in ['confidence', 'margin', 'entropy'])
        names.append(f'{name}_disagrees_fixed')
    features.extend([np.eye(raw.shape[1], dtype=np.float32)[raw.argmax(1)], structural])
    names.extend(f'raw_predicted_class_{i}' for i in range(raw.shape[1]))
    names.extend(['log_degree', 'neighbor_support_raw', 'neighbor_class_agrees_raw',
                  'neighbor_entropy', 'raw_neighbor_probability_overlap'])
    state = np.concatenate(features, 1).astype(np.float32)
    assert np.isfinite(state).all()
    return state, names


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'dataset.npz'
    if not cache.exists():
        archive = ROOT.parent / 'arxiv.zip'
        extracted = ROOT.parent / 'download'
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (extracted / name).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError('unsafe archive path')
            z.extractall(extracted)
        data = extracted / 'arxiv'
        splits = {k: pd.read_csv(data / f'split/time/{k}.csv.gz', header=None).values.ravel().astype(np.int64)
                  for k in ['train', 'valid', 'test']}
        x = pd.read_csv(data / 'raw/node-feat.csv.gz', header=None).values.astype(np.float32)
        # Label-free transductive standardization, matching the original arxiv MLP convention.
        std = x.std(0, ddof=1)
        std[std < 1e-6] = 1
        x = (x - x.mean(0)) / std
        y = pd.read_csv(data / 'raw/node-label.csv.gz', header=None).values.ravel().astype(np.int64)
        y[splits['test']] = -1
        edge = pd.read_csv(data / 'raw/edge.csv.gz', header=None).values
        adj = sparse.coo_matrix((np.ones(2 * len(edge), dtype=np.float32),
            (np.concatenate([edge[:, 0], edge[:, 1]]), np.concatenate([edge[:, 1], edge[:, 0]]))),
            shape=(len(x), len(x))).tocsr()
        adj.sum_duplicates()
        adj.data[:] = 1
        sparse.save_npz(ROOT / 'adjacency.npz', adj)
        assert len(x) == 169343 and len(splits['train']) == 90941
        assert len(splits['valid']) == 29799 and len(splits['test']) == 48603
        assert len(np.unique(np.concatenate(list(splits.values())))) == len(x)
        np.savez_compressed(cache, x=x, labels=y, **splits)
        (ROOT / 'dataset-manifest.json').write_text(json.dumps({
            'dataset': 'ogbn-arxiv', 'archive_sha256': digest(archive), 'num_classes': 40,
            'split_counts': {k: len(v) for k, v in splits.items()},
            'features': '128 official features, label-free standardization; no spectral/diffusion augmentation',
            'graph': 'binary, symmetric, coalesced citation graph', 'test_labels_masked': True}, indent=2) + '\n')
    with np.load(cache) as z:
        arrays = {k: z[k] for k in z.files}
    assert (arrays['labels'][arrays['test']] == -1).all()
    return arrays


def adjacency_operators():
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    adj = sparse.load_npz(ROOT / 'adjacency.npz')
    degree = np.asarray(adj.sum(1)).ravel()
    inverse = np.divide(1, degree, out=np.zeros_like(degree), where=degree > 0)
    gpu = csr_matrix(adj)
    d = cp.asarray(np.sqrt(inverse))
    return degree, GPUAdjacency(gpu.multiply(d[:, None]).multiply(d[None, :]).tocsr()), \
        GPUAdjacency(gpu.multiply(cp.asarray(inverse)[:, None]).tocsr())


def fit_model(x, y, train, valid, seed, path):
    if path.exists():
        return torch.load(path, map_location='cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = original_mlp()(x.shape[1], 200, 40, 3, .5, False).cuda()
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    tr, va = torch.from_numpy(train).cuda(), torch.from_numpy(valid).cuda()
    best_accuracy = -1
    best = None
    selected_epoch = None
    for epoch in range(1, 300):
        model.train()
        opt.zero_grad()
        loss = F.nll_loss(model(x[tr]), y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            score = float((model(x[va]).argmax(1) == y[va]).float().mean())
        if score > best_accuracy:
            best_accuracy, best, selected_epoch = score, copy.deepcopy(model.state_dict()), epoch
        if epoch % 100 == 0:
            print(f'{path.stem} epoch {epoch}: validation={score:.5f}', flush=True)
    model.load_state_dict(best)
    with torch.no_grad():
        raw = torch.cat([model(chunk).exp().cpu() for chunk in x.split(100000)])
    torch.save(best, path.with_suffix('.weights.pt'))
    path.with_suffix('.json').write_text(json.dumps({'seed': seed, 'best_validation': best_accuracy,
        'selected_epoch': selected_epoch, 'train_ids_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()), 'official_test_scored': False}, indent=2) + '\n')
    torch.save(raw, path)
    return raw


def bank_and_state(raw, labels, anchors, targets, degree, dad, da):
    import cupy as cp
    targets = torch.as_tensor(targets)
    anchors = torch.as_tensor(anchors)
    assert not np.intersect1d(anchors.numpy(), targets.numpy()).size
    split = {'train': anchors, 'valid': torch.empty(0, dtype=torch.long), 'test': torch.empty(0, dtype=torch.long)}
    masked = anchor_labels(labels, anchors)
    cs = original_cs()
    bank = []
    for name, (scale, alpha) in ACTIONS:
        _, value = cs.double_correlation_fixed(SimpleNamespace(y=masked[:, None]), raw, split,
            dad, 1., 50, da, alpha, 50, scale=scale, train_only=True, device='cuda', display=False)
        value = value[targets].clamp_min(0)
        value /= value.sum(1, keepdim=True).clamp_min(1e-12)
        bank.append(value.numpy())
        print(f'C&S {name}: {len(targets)} targets', flush=True)
    own = raw[targets].numpy()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        neighbor = cp.asnumpy((da.matrix @ cp.asarray(raw.numpy()))[cp.asarray(targets.numpy())])
    neighbor /= np.maximum(neighbor.sum(1, keepdims=True), 1e-12)
    structural = np.stack([np.log1p(degree[targets]), neighbor[np.arange(len(own)), own.argmax(1)],
        (neighbor.argmax(1) == own.argmax(1)).astype(np.float32),
        -(neighbor * np.log(np.maximum(neighbor, 1e-12))).sum(1), (neighbor * own).sum(1)], 1)
    bank = np.stack(bank)
    state, names = state_features(bank, own, structural)
    return bank, state, names


def prepare(seed):
    out = ROOT / f'seed-{seed}'
    if (out / 'manifest.json').exists():
        return
    out.mkdir(exist_ok=True)
    data = setup()
    x = torch.from_numpy(data['x']).cuda()
    y = torch.from_numpy(data['labels'])
    train, valid = data['train'], data['valid']
    raw = fit_model(x, y.cuda(), train, valid, seed, out / 'full.pt')
    degree, dad, da = adjacency_operators()
    bank_va, state_va, names = bank_and_state(raw, y, train, valid, degree, dad, da)
    n = len(train) + len(valid)
    bank = np.empty((5, n, 40), dtype=np.float32)
    state = np.empty((n, len(names)), dtype=np.float32)
    bank[:, len(train):], state[len(train):] = bank_va, state_va
    permutation = np.random.default_rng(20261029).permutation(len(train))
    for fold, positions in enumerate(np.array_split(permutation, 3)):
        anchors = train[np.setdiff1d(np.arange(len(train)), positions)]
        raw = fit_model(x, y.cuda(), anchors, valid, seed * 10 + 100 + fold, out / f'fold-{fold}.pt')
        p, s, _ = bank_and_state(raw, y, anchors, train[positions], degree, dad, da)
        bank[:, positions], state[positions] = p, s
        np.save(out / f'fold-{fold}-target-ids.npy', train[positions])
    ids = np.concatenate([train, valid])
    assert not np.intersect1d(ids, data['test']).size
    np.savez_compressed(out / 'development.npz', state=state, probabilities=bank, labels=data['labels'][ids],
        train=np.arange(len(train)), selection=np.arange(len(train), n), original_node_ids=ids)
    manifest = {'protocol': 'cross_dataset_oof_v1', 'dataset': 'ogbn-arxiv', 'base_seed': seed,
        'train_rows': len(train), 'selection_rows': len(valid), 'actions': ACTIONS,
        'feature_names': names, 'num_classes': 40, 'official_test_scored': False,
        'excluded_diagnostic_rows': 0, 'source_sha256': digest(Path(__file__)),
        'baseline_validation': {name: float((bank_va[a].argmax(1) == data['labels'][valid]).mean())
                                for a, (name, _) in enumerate(ACTIONS)}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['baseline_validation']), flush=True)


def final_test():
    """Freeze all development outputs before reading official test targets once."""
    final = ROOT / 'final-test'
    final.mkdir(exist_ok=True)
    if (final / 'TEST_ACCESS_STARTED').exists():
        raise RuntimeError('Final test access already started; preserve frozen evaluation')
    if not (ROOT / 'DEVELOPMENT_COMPLETE').exists():
        raise RuntimeError('Development not complete')
    paths = [Path(__file__), Path(__file__).with_name('train_cross_dataset.py'),
             Path(__file__).with_name('CROSS_DATASET_PROTOCOL.md'),
             Path(__file__).with_name('prepare_small_cs.py'), Path(__file__).with_name('mlp_support.py'),
             Path(__file__).with_name('train_v2.py'), Path(__file__).with_name('train_pilot.py'),
             ROOT / 'dataset.npz', ROOT / 'adjacency.npz']
    selected = {}
    for seed in range(3):
        policy = ROOT / f'policy-{seed}'
        summary = json.loads((policy / 'summary.json').read_text())
        selected[str(seed)] = [name for name, _ in ACTIONS].index(summary['best_global_action'])
        paths.extend([ROOT / f'seed-{seed}/full.pt', ROOT / f'seed-{seed}/manifest.json',
                      policy / 'summary.json', policy / 'config.json', policy / 'normalization.npz'])
        paths.extend(policy / f'{method}-{actor_seed}-actor.pt'
                     for method in ['ppo', 'supervised', 'expected_reward'] for actor_seed in range(20, 25))
    lock = {'artifacts': {str(path): digest(path) for path in paths},
            'selected_global': selected, 'actions': ACTIONS,
            'prior_exposure': 'Products test results informed the frozen follow-up design; no arxiv test accuracy used for development.'}
    (final / 'lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    (final / 'TEST_ACCESS_STARTED').write_text(digest(final / 'lock.json') + '\n')
    data = setup()
    # This is the only stage where arxiv test labels are used.
    truth = pd.read_csv(ROOT.parent / 'download/arxiv/raw/node-label.csv.gz', header=None).values.ravel()[data['test']]
    degree, dad, da = adjacency_operators()
    records = []
    for seed in range(3):
        policy = ROOT / f'policy-{seed}'
        raw = torch.load(ROOT / f'seed-{seed}/full.pt', map_location='cpu')
        bank, state, _ = bank_and_state(raw, torch.from_numpy(data['labels']), data['train'], data['test'], degree, dad, da)
        correct = bank.argmax(2).T == truth[:, None]
        fixed = float(correct[:, 1].mean())
        with np.load(policy / 'normalization.npz') as z:
            state = (state - z['mean']) / z['scale']
        actors = []
        for method in ['ppo', 'supervised', 'expected_reward']:
            for actor_seed in range(20, 25):
                actor = Actor(state.shape[1], 32, 5)
                actor.load_state_dict(torch.load(policy / f'{method}-{actor_seed}-actor.pt', map_location='cpu'))
                actor.eval()
                with torch.no_grad():
                    choices = actor(torch.from_numpy(state)).argmax(1).numpy()
                actors.append({'method': method, 'actor_seed': actor_seed,
                               'accuracy': float(correct[np.arange(len(choices)), choices].mean())})
                np.save(final / f'base-{seed}-{method}-{actor_seed}-choices.npy', choices.astype(np.int8))
        record = {'base_seed': seed, 'fixed_cs_accuracy': fixed,
                  'validation_selected_global_accuracy': float(correct[:, selected[str(seed)]].mean()),
                  'actors': actors}
        for method in ['ppo', 'supervised', 'expected_reward']:
            record[f'{method}_mean_accuracy'] = float(np.mean([a['accuracy'] for a in actors if a['method'] == method]))
        record['ppo_gain_pp'] = 100 * (record['ppo_mean_accuracy'] - fixed)
        records.append(record)
        (final / 'per-seed.json').write_text(json.dumps(records, indent=2) + '\n')
    summary = {'dataset': 'ogbn-arxiv', 'base_seeds': 3, 'actors_per_base': 5,
               'test_nodes': len(truth), 'official_test_scored': True, 'per_seed': records,
               'lock_sha256': digest(final / 'lock.json')}
    for name in ['fixed_cs_accuracy', 'ppo_mean_accuracy', 'supervised_mean_accuracy',
                 'expected_reward_mean_accuracy', 'validation_selected_global_accuracy', 'ppo_gain_pp']:
        values = [r[name] for r in records]
        summary[name] = {'mean': float(np.mean(values)), 'sample_std_across_base_seeds': float(np.std(values, ddof=1))}
    (final / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (final / 'COMPLETE').write_text('Locked arxiv final evaluation complete\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_seed'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare', 'final'], required=True)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.phase == 'prepare':
        if args.seed not in range(3):
            raise ValueError('Expected base seeds 0, 1, 2')
        setup()
        prepare(args.seed)
    else:
        final_test()
