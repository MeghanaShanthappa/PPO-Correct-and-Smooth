"""Summarize completed MLP/C&S and PPO validation runs without test scoring."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/var/tmp/mshanthappa/rl-cs/mlp')
    p.add_argument('--report',default='/home/mshanthappa/ogbn-products-rlcs/RESULTS_MLP.md')
    args=p.parse_args(); root=Path(args.root)
    rows=[]
    for seed in range(10):
        base=root/f'seed-{seed}/baseline-validation.json'
        if not base.exists(): continue
        try:
            r={'base_seed':seed,**json.loads(base.read_text())}
            policy=root/f'policy-{seed}/summary.json'
            if policy.exists():
                s=json.loads(policy.read_text())
                assert s['selection_nodes']==39323 and not s['official_test_scored']
                r['ppo_mean_validation']=s['methods']['ppo']['mean_accuracy']
                r['supervised_mean_validation']=s['methods']['supervised']['mean_accuracy']
                r['ppo_gain_pp']=100*(r['ppo_mean_validation']-r['fixed_cs'])
            rows.append(r)
        except json.JSONDecodeError: continue
    paired=[r for r in rows if 'ppo_mean_validation' in r]
    result={'published_reference':{'test_accuracy':.8418,'validation_accuracy':.9147},
        'planned_base_seeds':10,'completed_baseline_scores':len(rows),'completed_paired_comparisons':len(paired),
        'evaluation':'full official validation; checkpoint-selected, not untouched',
        'official_test_scored':False,'per_seed':rows}
    if paired:
        result['paired_mean_gain_pp']=float(np.mean([r['ppo_gain_pp'] for r in paired]))
        result['paired_gain_sample_std_pp']=float(np.std([r['ppo_gain_pp'] for r in paired],ddof=1)) if len(paired)>1 else None
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Original MLP + C&S and PPO reproduction','',
        f'Completed baseline scores: {len(rows)}/10. Completed paired PPO comparisons: {len(paired)}/10.',
        '', 'Every local score below uses the **full 39,323-node official validation set**. '
        'That set selects checkpoints, so these are development comparisons, not independent '
        'test results. The published reference is 91.47% validation and 84.18% test; '
        'the latter has not been reproduced or scored in this run.','',
        '| Base seed | Raw MLP validation | Original C&S validation | PPO mean validation | PPO minus C&S (pp) |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        a=f"{100*r['ppo_mean_validation']:.4f}%" if 'ppo_mean_validation' in r else 'pending'
        gain=f"{r['ppo_gain_pp']:+.4f}" if 'ppo_gain_pp' in r else 'pending'
        lines.append(f"| {r['base_seed']} | {100*r['raw']:.4f}% | {100*r['fixed_cs']:.4f}% | {a} | {gain} |")
    lines.extend(['','The model is the original 96,247-parameter architecture with default OGB '
        'features and graph-only spectral features. The spectral eigensolver is a disclosed '
        'GPU port of the original Julia operator with residual/orthogonality checks. '
        'C&S uses the original 50/50-step DAD/DA implementation.','',
        'PPO training rewards come from three-fold out-of-fold predictions of official '
        'training nodes. Each target fold is excluded from classifier training and C&S '
        'anchors. No validation labels are used in policy rewards. The same training '
        'label budget is available to the fixed baseline and policy. Validation labels '
        'still select base and policy checkpoints.','',
        'Five actor seeds are averaged within each base-model run; they do not replace '
        'independent base-model seeds. Report all ten base runs before claiming the '
        'published mean and standard deviation are reproduced. No official test '
        'improvement is established by validation results alone.','',
        'Frozen setup: [MLP_PROTOCOL.md](MLP_PROTOCOL.md). Live log: '
        '[mlp-experiment.log](mlp-experiment.log).',''])
    Path(args.report).write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='per_seed'},indent=2))


if __name__=='__main__': main()
