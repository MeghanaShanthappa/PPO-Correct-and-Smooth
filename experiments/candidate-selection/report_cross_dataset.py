"""Summarize completed datasets without accessing any target labels."""
import json
from pathlib import Path
import numpy as np

BASE = Path('/var/tmp/mshanthappa/rl-cs')
rows = []
details = {}
for dataset, root in [('ogbn-products', BASE / 'small-cs-v1'),
                      ('ogbn-arxiv', BASE / 'cross-dataset-v1/ogbn-arxiv')]:
    records = [json.loads(p.read_text()) for p in sorted(root.glob('policy-*/summary.json'))]
    if not records:
        rows.append(f'| {dataset} | 0/3 | pending | pending | pending |')
        continue
    fixed = np.array([r['global_actions']['fixed_cs'] for r in records])
    ppo = np.array([r['methods']['ppo']['mean_accuracy'] for r in records])
    best = np.array([r['best_global_selection_accuracy'] for r in records])
    details[dataset] = {'completed_base_seeds': len(records), 'fixed_mean': float(fixed.mean()),
        'ppo_mean': float(ppo.mean()), 'ppo_gain_pp': (100 * (ppo - fixed)).tolist(),
        'mean_gain_over_best_global_pp': float(100 * (ppo - best).mean()),
        'gain_sample_std_pp': float(np.std(100 * (ppo - fixed), ddof=1)) if len(records) > 1 else None}
    rows.append(f'| {dataset} | {len(records)}/3 | {100*fixed.mean():.3f}% | {100*ppo.mean():.3f}% | {100*(ppo-fixed).mean():+.3f} |')
lines = ['# Cross-dataset local C&S versus PPO', '',
         'Validation is used for checkpoint selection; these scores are development results.', '',
         '| Dataset | Base seeds | Fixed C&S | PPO C&S | Gain (percentage points) |',
         '|---|---:|---:|---:|---:|', *rows, '',
         'Within each base seed, PPO accuracy averages five actor seeds. Base models are trained independently per dataset; this tests method transfer, not zero-shot policy transfer.', '',
         'Products uses its existing spectral-augmented MLP. Arxiv uses standardized official 128-dimensional features without added spectral/diffusion features. This is not an arxiv leaderboard-baseline reproduction.', '']
final = BASE / 'cross-dataset-v1/ogbn-arxiv/final-test/summary.json'
if final.exists():
    r = json.loads(final.read_text())
    details['arxiv_final_test'] = r
    lines += ['## Frozen arxiv final test', '',
        f"Fixed C&S: {100*r['fixed_cs_accuracy']['mean']:.3f}%. PPO C&S: {100*r['ppo_mean_accuracy']['mean']:.3f}%.",
        f"Paired mean gain: {r['ppo_gain_pp']['mean']:+.3f} percentage points across three base seeds.",
        f"Validation-best fixed action: {100*r['validation_selected_global_accuracy']['mean']:.3f}%.",
        'This completed result must be reported whether positive or negative. No settings change after test access.', '']
else:
    lines += ['Arxiv final-test evaluation is pending. The settings and models will be locked first.', '']
lines += ['The earlier, different Products PPO experiment lost 2.467 test points; that result remains part of the record. The new Products small-adjustment experiment has no test score.', '']
Path('/home/mshanthappa/ogbn-products-rlcs/RESULTS_CROSS_DATASET.md').write_text('\n'.join(lines))
out = BASE / 'cross-dataset-v1'
out.mkdir(exist_ok=True)
(out / 'summary.json').write_text(json.dumps(details, indent=2) + '\n')
print('\n'.join(lines), flush=True)
