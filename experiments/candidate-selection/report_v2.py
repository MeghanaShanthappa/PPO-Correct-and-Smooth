"""Verify saved V2 choices and write a development-only result report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    root=Path(args.run)
    summary=json.loads((root/'summary.json').read_text())
    config=json.loads((root/'config.json').read_text())
    records=json.loads((root/'runs.json').read_text())
    data=Path(config['data'])/'development.npz'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==config['data_sha256']
    with np.load(data,allow_pickle=False) as z:
        selection=z['selection']; probabilities=z['probabilities'][:,selection]
        labels=z['labels'][selection]
    correct=probabilities.argmax(2).T==labels[:,None]
    assert len(records)==20
    for r in records:
        choice=np.load(root/f"{r['method']}-{r['seed']}-choices.npy",allow_pickle=False)
        selected=correct[np.arange(len(selection)),choice]
        assert float(selected.mean())==r['selection']['accuracy']
        assert int((selected&~correct[:,1]).sum())==r['selection']['fixed_errors']
        assert int((~selected&correct[:,1]).sum())==r['selection']['introduced_errors']
    integrity={'all_20_saved_choice_vectors_verified':True,'data_hash_verified':True,
        'selection_nodes':len(selection),'diagnostic_scored':False,'official_test_scored':False}
    (root/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    lines=['# V2 PPO-controlled C&S: development results','',
        'This run uses one GraDBERT/GIANT base-model seed and five controller seeds. '
        'All numbers below are accuracy on the 7,865-node **selection set**, used '
        'to select checkpoints. They are not held-out accuracy or official test results. '
        'The earlier diagnostic cohort was excluded from the data package.','',
        '## Fixed action bank','', '| Action | Selection accuracy |','|---|---:|']
    for name,accuracy in summary['global_actions'].items():
        lines.append(f'| {name} | {100*accuracy:.4f}% |')
    lines.extend(['','## Controller comparison','',
        '| Controller | Mean selection accuracy | Gain over fixed C&S (pp) | Gain over best global action (pp) |',
        '|---|---:|---:|---:|'])
    for name,m in summary['methods'].items():
        lines.append(f"| {name} | {100*m['mean_accuracy']:.4f}% | {m['gain_over_fixed_cs_pp']:+.4f} | {m['gain_over_best_global_pp']:+.4f} |")
    ppo=summary['methods']['ppo']['mean_accuracy']
    warm=summary['methods']['warm_start']['mean_accuracy']
    lines.extend(['',f'PPO minus the supervised warm-start mean: **{100*(ppo-warm):+.4f} percentage points**.',
        '',f"The best global action on this same selection set is `{summary['best_global_action']}`. "
        'Its selection advantage is not an independent estimate either. Gains over the original '
        'fixed C&S setting alone cannot establish that PPO is responsible; compare both global '
        'action tuning and supervised training.','',
        '## What changed','',
        'The controller has six precomputed actions, 94 label-free state features, a 32-unit '
        'hidden layer (3,238 actor parameters), supervised initialization, and a dense '
        'training reward combining correctness and true-class log probability. PPO uses '
        'an exact action-independent baseline computed from training action outcomes. '
        'The same architecture and data support the supervised and expected-reward controls.','',
        'The original exporter used `topk` to choose the predicted class. Six allowed development '
        'rows had exact tied top scores and differed from standard `argmax`. V2 consistently '
        'uses `argmax` across every method; the old results are preserved with a caveat. '
        'No PPO credit should be assigned to changing tie-breaking.','',
        '## Limits and verification','',
        'All 20 saved choice vectors were checked against the reported accuracies and '
        'corrected/introduced mistake counts. The development data hash matches its training '
        'configuration. Eleven code tests passed before the experiment completed.','',
        'The base model used full-validation checkpoint selection. These development results '
        'therefore do not establish generalization, a robust PPO improvement, a leaderboard '
        'score, or a paper-level result. Future evaluation must freeze the method and include '
        'additional base-model seeds and appropriately isolated targets. All six global '
        'propagation candidates must be computed before selection; this implementation '
        'does not reduce propagation cost.','',
        f'Raw run records: `{root}`. The frozen protocol is [V2_PROTOCOL.md](V2_PROTOCOL.md).',''])
    Path(args.report).write_text('\n'.join(lines))
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
