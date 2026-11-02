"""Exploratory PPO/SFT comparison using historical validation-only action exports.

This pilot chooses between cached raw and C&S predictions; it does not rerun
propagation, reproduce a leaderboard model, or load official test labels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ACTIONS = ('raw', 'locked_cs_c035_s060_z20')
FIELDS = ('prediction', 'confidence', 'margin', 'entropy')


def load_validation(root, action_names=ACTIONS):
    root = Path(root)
    metadata = [json.loads((root / f'{name}.json').read_text()) for name in action_names]
    first = metadata[0]
    train, valid = first['train_rows'], first['validation_rows']
    if (train, valid) != (196615, 39323):
        raise ValueError('unexpected OGB-Products export dimensions')
    arrays = []
    for name, meta in zip(action_names, metadata):
        if meta['test_rows_exported'] != 0 or any(meta[k] != first[k] for k in ('train_rows', 'validation_rows', 'seeds')):
            raise ValueError('inconsistent or test-containing action export')
        with np.load(root / f'{name}.npz', allow_pickle=False) as archive:
            if any(archive[k].shape != (train + valid,) for k in FIELDS):
                raise ValueError('unexpected action shape')
            arrays.append({k: archive[k][train:].astype(np.float32) for k in FIELDS})
    labels = np.load(root / 'train_valid_labels.npy', allow_pickle=False)
    if labels.shape != (train + valid,):
        raise ValueError('unexpected label shape')
    labels = labels[train:]
    if np.any((labels < 0) | (labels >= 47)):
        raise ValueError('invalid class labels')
    columns = [a[k] for a in arrays for k in FIELDS[1:]]
    for k in FIELDS[1:]:
        delta = arrays[1][k] - arrays[0][k]
        columns.extend([delta, np.abs(delta)])
    predictions = np.stack([a['prediction'].astype(np.int64) for a in arrays], axis=1)
    columns.append((predictions[:, 0] != predictions[:, 1]).astype(np.float32))
    context = np.stack(columns, axis=1)
    if not np.isfinite(context).all():
        raise ValueError('nonfinite context')
    rewards = (predictions == labels[:, None]).astype(np.float32)
    return context, rewards, predictions, metadata


def split_rows(count, seed):
    """Index-only split; neither labels nor action outcomes influence membership."""
    rows = np.random.default_rng(seed).permutation(count)
    n, m = int(.6 * count), int(.8 * count)
    return rows[:n], rows[n:m], rows[m:]


class Actor(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([0., 1.]))

    def forward(self, x):
        return self.net(x)


def clipped_objective(log_prob, old_log_prob, advantage, clip):
    ratio = (log_prob - old_log_prob).exp()
    return torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)


@torch.no_grad()
def choices(actor, x):
    return actor(x).argmax(1).numpy()


def score(rewards, selected):
    return float(rewards[np.arange(len(selected)), selected].mean())


def train(method, seed, x, rewards, predictions, train_ids, select_ids, args):
    torch.manual_seed(seed)
    # Identical predictions cannot change accuracy under either action. Train
    # both methods on the same disagreement cohort; evaluate every held-out row.
    active = train_ids[predictions[train_ids, 0] != predictions[train_ids, 1]]
    if len(active) == 0:
        raise ValueError('no training action disagreements')
    actor = Actor(x.shape[1], args.hidden)
    critic = nn.Sequential(nn.Linear(x.shape[1], args.hidden), nn.Tanh(), nn.Linear(args.hidden, 1))
    parameters = list(actor.parameters()) + (list(critic.parameters()) if method == 'ppo' else [])
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    tx, tr = x[active], torch.from_numpy(rewards[active])
    best_state, best_epoch = copy.deepcopy(actor.state_dict()), 0
    best_accuracy = score(rewards[select_ids], choices(actor, x[select_ids]))
    curve = []
    started = time.perf_counter()
    updates = 0
    for epoch in range(1, args.epochs + 1):
        if method == 'ppo':
            with torch.no_grad():
                dist = Categorical(logits=actor(tx))
                action = dist.sample()
                old_log = dist.log_prob(action)
                # Reward relative to the fixed C&S action, using training labels.
                outcome = tr[torch.arange(len(active)), action] - tr[:, 1]
                advantage = outcome - critic(tx).squeeze(1)
                advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        for _ in range(args.update_epochs):
            order = torch.randperm(len(active))
            for batch in order.split(args.batch_size):
                logits = actor(tx[batch])
                if method == 'ppo':
                    dist = Categorical(logits=logits)
                    objective = clipped_objective(dist.log_prob(action[batch]), old_log[batch], advantage[batch], args.clip)
                    value_loss = (critic(tx[batch]).squeeze(1) - outcome[batch]).square().mean()
                    loss = -objective.mean() - args.entropy * dist.entropy().mean() + .5 * value_loss
                else:
                    useful = tr[batch, 0] != tr[batch, 1]
                    if not useful.any():
                        continue
                    loss = nn.functional.cross_entropy(logits[useful], tr[batch][useful].argmax(1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, .5)
                optimizer.step()
                updates += 1
        accuracy = score(rewards[select_ids], choices(actor, x[select_ids]))
        curve.append({'epoch': epoch, 'selection_accuracy': accuracy})
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = copy.deepcopy(actor.state_dict())
    actor.load_state_dict(best_state)
    return actor, {'seed': seed, 'selected_epoch': best_epoch, 'selection_accuracy': best_accuracy,
                   'training_disagreement_rows': len(active), 'optimizer_updates': updates,
                   'seconds': time.perf_counter() - started, 'curve': curve}


def heldout_metrics(rewards, selected):
    correct = rewards[np.arange(len(selected)), selected].astype(bool)
    base = rewards[:, 1].astype(bool)
    return {'accuracy': float(correct.mean()), 'baseline_cs_accuracy': float(base.mean()),
            'gain_pp': 100 * float((correct.astype(float) - base).mean()),
            'fixed_errors': int((correct & ~base).sum()), 'introduced_errors': int((~correct & base).sum()),
            'raw_selected': int((selected == 0).sum()), 'rows': len(selected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--actions', nargs=2, default=list(ACTIONS))
    p.add_argument('--split-seed', type=int, default=20261028)
    p.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--entropy', type=float, default=.01)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    context, rewards, predictions, metadata = load_validation(args.candidates, args.actions)
    train_ids, select_ids, holdout_ids = split_rows(len(context), args.split_seed)
    # Fit preprocessing on policy-training rows only.
    mean, scale = context[train_ids].mean(0), context[train_ids].std(0)
    scale[scale < 1e-6] = 1
    x = torch.from_numpy((context - mean) / scale)
    np.savez_compressed(out / 'split-and-normalization.npz', train=train_ids, selection=select_ids,
                        diagnostic_holdout=holdout_ids, mean=mean, scale=scale)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(args.candidates).iterdir()
              if f.name in {'train_valid_labels.npy', *[f'{a}.{ext}' for a in args.actions for ext in ('json', 'npz')]}}
    scope = ('Author-source GIANT/GraDBERT-KD SAGN/SLE seed-0 candidate actions; exploratory official-validation resplit; not an official test score.'
             if metadata[0].get('feature_stack', '').startswith('authors GIANT') else
             'Historical hybrid-stack ensemble, exploratory official-validation resplit; not exact leaderboard reproduction or untouched validation.')
    config = {**vars(args), 'input_sha256': hashes, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_metadata': metadata, 'torch': torch.__version__, 'numpy': np.__version__,
              'scope': scope}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    results = []
    for method in ('supervised', 'ppo'):
        for seed in args.seeds:
            actor, record = train(method, seed, x, rewards, predictions, train_ids, select_ids, args)
            record['method'] = method
            # The diagnostic cohort is scored only after this seed's checkpoint
            # is selected; it is never used in optimization or checkpoint choice.
            selected = choices(actor, x[holdout_ids])
            record['diagnostic_holdout'] = heldout_metrics(rewards[holdout_ids], selected)
            np.save(out / f'{method}-{seed}-choices.npy', selected, allow_pickle=False)
            torch.save(actor.state_dict(), out / f'{method}-{seed}-actor.pt')
            results.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'curve'}), flush=True)
    rr = rewards[holdout_ids]
    summary = {'scope': config['scope'], 'official_test_labels_loaded_or_scored': False,
               'policy_split_sizes': [len(train_ids), len(select_ids), len(holdout_ids)],
               'baseline_raw_accuracy': float(rr[:, 0].mean()), 'baseline_cs_accuracy': float(rr[:, 1].mean()),
               'oracle_accuracy_upper_bound': float(rr.max(1).mean()), 'methods': {}}
    for method in ('supervised', 'ppo'):
        group = [r for r in results if r['method'] == method]
        accuracy = [r['diagnostic_holdout']['accuracy'] for r in group]
        summary['methods'][method] = {'mean_accuracy': float(np.mean(accuracy)),
            'sample_std': float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else None,
            'mean_gain_pp': 100 * (float(np.mean(accuracy)) - summary['baseline_cs_accuracy'])}
    (out / 'runs.json').write_text(json.dumps(results, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
