"""Post-hoc diagnosis of frozen runs; no training or new test accuracy scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from train_v2 import Actor

ROOT = Path('/var/tmp/mshanthappa/rl-cs/mlp')
NAMES = ['raw', 'fixed_cs', 'scale5', 'scale20', 'smooth06', 'smooth09']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(a):
    return {'mean': float(np.mean(a)), 'quantiles_0_10_50_90_100':
            np.quantile(a, [0, .1, .5, .9, 1]).tolist()}


def main():
    torch.set_num_threads(4)
    final = ROOT / 'final-test'
    lock = json.loads((final / 'lock.json').read_text())
    mismatches = [p for p, value in lock['artifacts'].items() if sha(p) != value]
    assert not mismatches, mismatches
    result = {'scope': 'post-hoc diagnosis; no retraining or new official-test accuracy evaluation',
              'frozen_artifacts_verified': len(lock['artifacts']), 'per_seed': []}
    test_records = json.loads((final / 'per-seed.json').read_text())
    for seed in range(10):
        folder = ROOT / f'seed-{seed}'
        with np.load(folder / 'development.npz') as z:
            probs, labels, tr, va = (z[k] for k in ['probabilities', 'labels', 'train', 'selection'])
            correct = probs.argmax(2).T == labels[:, None]
            record = {'base_seed': seed,
                      'oof_training_action_accuracy': dict(zip(NAMES, correct[tr].mean(0).tolist())),
                      'validation_action_accuracy': dict(zip(NAMES, correct[va].mean(0).tolist())),
                      'validation_action_disagreement_with_fixed': dict(zip(NAMES, (probs[:, va].argmax(2) != probs[1, va].argmax(1)).mean(1).tolist())),
                      'policies': {}}
            for method in ['ppo', 'supervised']:
                valid_counts, test_counts = [], []
                for actor_seed in range(20, 25):
                    vc = np.load(ROOT / f'policy-{seed}/{method}-{actor_seed}-choices.npy')
                    tc = np.load(final / f'base-{seed}-{method}-actor-{actor_seed}-choices.npy')
                    valid_counts.append(np.bincount(vc, minlength=6) / len(vc))
                    test_counts.append(np.bincount(tc, minlength=6) / len(tc))
                record['policies'][method] = {
                    'validation_action_fraction': dict(zip(NAMES, np.mean(valid_counts, 0).tolist())),
                    'test_action_fraction': dict(zip(NAMES, np.mean(test_counts, 0).tolist()))}
            if seed == 0:
                state = z['state']
                manifest = json.loads((folder / 'manifest.json').read_text())
                result['oof_vs_validation_state_seed0'] = {
                    name: {'oof': describe(state[tr, i]), 'validation': describe(state[va, i])}
                    for i, name in enumerate(manifest['feature_names'])
                    if name in ['raw_confidence', 'fixed_cs_confidence', 'log_degree', 'neighbor_support_raw']}
                with np.load(ROOT / 'policy-0/normalization.npz') as norm:
                    normstate = (state - norm['mean']) / norm['scale']
                actor = Actor(state.shape[1], 32, 6)
                actor.load_state_dict(torch.load(ROOT / 'policy-0/ppo-20-actor.pt', map_location='cpu'))
                with torch.no_grad():
                    choices = torch.cat([actor(torch.from_numpy(b)).argmax(1) for b in np.array_split(normstate, 10)]).numpy()
                saved = np.load(ROOT / 'policy-0/ppo-20-choices.npy')
                assert np.array_equal(choices[va], saved), 'validation inference replay mismatch'
                validation_correct_fixed = correct[va, 1].copy()
                validation_correct_policy = correct[va, saved].copy()
                result['validation_replay_exact_seed0_actor20'] = True
                record['ppo20_oof_accuracy_posthoc'] = float(correct[tr, choices[tr]].mean())
                record['ppo20_oof_gain_pp_posthoc'] = 100 * float((correct[tr, choices[tr]].astype(float) - correct[tr, 1]).mean())
                # Explicit train/validation node identity and OOF anchor exclusion checks.
                ids = z['original_node_ids']
                folds = [np.load(folder / f'fold-{f}-target-node-ids.npy') for f in range(3)]
                assert np.array_equal(np.sort(np.concatenate(folds)), np.sort(ids[tr]))
                assert not np.intersect1d(ids[tr], ids[va]).size
                result['split_identity_checks_seed0'] = True
            record['frozen_test_gain_pp'] = test_records[seed]['paired_gain_pp']
            result['per_seed'].append(record)
        print(f'Analyzed seed {seed}', flush=True)
    with np.load(ROOT / 'spectral/split-ids.npz') as z:
        splits = {k: z[k] for k in z.files}
    adjacency = sparse.load_npz(ROOT / 'spectral/binary-adjacency.npz')
    degree = np.asarray(adjacency.sum(1)).ravel()
    anchors = np.zeros(adjacency.shape[0], dtype=np.float32)
    anchors[splits['train']] = 1
    counts = adjacency @ anchors
    fraction = np.divide(counts, degree, out=np.zeros_like(counts), where=degree > 0)
    result['validation_degree_slices_seed0_actor20'] = []
    vd = degree[splits['valid']]
    for low, high in [(0, 10), (10, 30), (30, 100), (100, float('inf'))]:
        mask = (vd >= low) & (vd < high)
        result['validation_degree_slices_seed0_actor20'].append({
            'degree_range': [low, high if np.isfinite(high) else None],
            'nodes': int(mask.sum()),
            'fixed_accuracy': float(validation_correct_fixed[mask].mean()),
            'ppo_accuracy': float(validation_correct_policy[mask].mean()),
            'gain_pp': 100 * float((validation_correct_policy[mask].astype(float) - validation_correct_fixed[mask]).mean())})
    raw = torch.load(ROOT / 'seed-0/full.pt', map_location='cpu').numpy()
    result['label_free_split_shift_seed0'] = {}
    for key, ids in splits.items():
        values = raw[ids]
        result['label_free_split_shift_seed0'][key] = {
            'nodes': len(ids), 'degree': describe(degree[ids]),
            'train_neighbor_fraction': describe(fraction[ids]),
            'zero_train_neighbors_fraction': float((counts[ids] == 0).mean()),
            'raw_confidence': describe(values.max(1)),
            'raw_predicted_class_fraction': (np.bincount(values.argmax(1), minlength=47) / len(ids)).tolist()}
    result['all_ten_test_gains_negative'] = all(r['paired_gain_pp'] < 0 for r in test_records)
    out = ROOT / 'diagnosis'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'frozen_artifacts_verified': result['frozen_artifacts_verified'],
                      'label_free_split_shift_seed0': result['label_free_split_shift_seed0']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
