"""Locked final paired official-test scoring after all ten seed runs complete.

No training, checkpoint selection or hyperparameter changes occur here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from scipy import sparse
import torch

from run_mlp_seed import action_bank,NAMES
from prepare_v2 import state_from_probabilities
from train_v2 import Actor


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024**2): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='/var/tmp/mshanthappa/rl-cs/mlp')
    args=p.parse_args(); root=Path(args.root)
    if not (root/'COMPLETE').exists(): raise RuntimeError('all ten development runs must finish first')
    final=root/'final-test'; final.mkdir(exist_ok=True)
    if (final/'TEST_ACCESS_STARTED').exists(): raise RuntimeError('preserve the one-time final evaluation; do not rescore')
    artifacts={}; selected_global={}
    for seed in range(10):
        validation=json.loads((root/f'seed-{seed}/baseline-validation.json').read_text())
        selected_global[str(seed)]=int(np.argmax([validation[name] for name in NAMES]))
        for path in [root/f'seed-{seed}/manifest.json',root/f'seed-{seed}/full.pt',
                     root/f'seed-{seed}/baseline-validation.json',
                     root/f'policy-{seed}/config.json',root/f'policy-{seed}/normalization.npz',
                     *[root/f'policy-{seed}/{m}-{a}-actor.pt' for m in ['ppo','supervised'] for a in range(20,25)]]:
            artifacts[str(path)]=digest(path)
    for name in ['MLP_PROTOCOL.md','run_mlp_seed.py','mlp_support.py','train_v2.py','evaluate_mlp_test_once.py']:
        path=Path('/home/mshanthappa/ogbn-products-rlcs')/name
        artifacts[str(path)]=digest(path)
    lock={'artifacts':artifacts,'selected_global_actions':selected_global,
          'scope':'frozen ten-seed paired MLP/C&S/PPO official-test evaluation',
          'prior_test_exposure':'Published Products references and earlier unrelated Products experiments were known; this does not erase that history.'}
    (final/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    (final/'TEST_ACCESS_STARTED').write_text(digest(final/'lock.json')+'\n')
    torch.set_num_threads(8)
    data=DglNodePropPredDataset(name='ogbn-products',root='/var/tmp/mshanthappa/rl-cs/ogb')
    _,labels=data[0]; labels=labels.squeeze()
    split=data.get_idx_split(); train,test=split['train'],split['test']
    truth=labels[test].numpy()
    adj=sparse.load_npz(root/'spectral/binary-adjacency.npz').astype(np.float32)
    degree=np.asarray(adj.sum(1)).reshape(-1)
    inverse=np.zeros_like(degree); np.divide(1,degree,out=inverse,where=degree>0)
    gpu=csr_matrix(adj); d=cp.asarray(np.sqrt(inverse))
    from mlp_support import GPUAdjacency
    dad=GPUAdjacency(gpu.multiply(d[:,None]).multiply(d[None,:]).tocsr())
    da=GPUAdjacency(gpu.multiply(cp.asarray(inverse)[:,None]).tocsr())
    results=[]
    for seed in range(10):
        print(f'Final locked test scoring: base seed {seed}',flush=True)
        raw=torch.load(root/f'seed-{seed}/full.pt',map_location='cpu')
        bank,own,neighbor=action_bank(raw,labels,train,dad,da,test)
        structural=np.stack([np.log1p(degree[test]),neighbor[np.arange(len(test)),own.argmax(1)],
            (neighbor.argmax(1)==own.argmax(1)).astype(np.float32),
            -(neighbor*np.log(np.maximum(neighbor,1e-12))).sum(1),(neighbor*own).sum(1)],1)
        state,_=state_from_probabilities(bank,structural,NAMES)
        predictions=bank.argmax(2).T
        correct=predictions==truth[:,None]
        fixed=float(correct[:,1].mean())
        policy=root/f'policy-{seed}'
        with np.load(policy/'normalization.npz') as z: mean,scale=z['mean'],z['scale']
        state=(state-mean)/scale
        actor_results=[]
        for method,a in [(m,a) for m in ['ppo','supervised'] for a in range(20,25)]:
            actor=Actor(state.shape[1],32,6).cuda()
            actor.load_state_dict(torch.load(policy/f'{method}-{a}-actor.pt',map_location='cuda'))
            actor.eval(); selected=[]
            with torch.no_grad():
                for start in range(0,len(state),50000):
                    selected.append(actor(torch.from_numpy(state[start:start+50000]).cuda()).argmax(1).cpu().numpy())
            choice=np.concatenate(selected)
            score=float(correct[np.arange(len(choice)),choice].mean())
            actor_results.append({'method':method,'actor_seed':a,'accuracy':score,'gain_pp':100*(score-fixed)})
            np.save(final/f'base-{seed}-{method}-actor-{a}-choices.npy',choice.astype(np.int8),allow_pickle=False)
        record={'base_seed':seed,'test_nodes':len(test),'raw_accuracy':float(correct[:,0].mean()),
            'fixed_cs_accuracy':fixed,
            'validation_selected_global_action':NAMES[selected_global[str(seed)]],
            'validation_selected_global_accuracy':float(correct[:,selected_global[str(seed)]].mean()),
            'ppo_actor_mean_accuracy':float(np.mean([r['accuracy'] for r in actor_results if r['method']=='ppo'])),
            'supervised_actor_mean_accuracy':float(np.mean([r['accuracy'] for r in actor_results if r['method']=='supervised'])),
            'actor_results':actor_results}
        record['paired_gain_pp']=100*(record['ppo_actor_mean_accuracy']-fixed)
        results.append(record)
        (final/'per-seed.json').write_text(json.dumps(results,indent=2)+'\n')
        del raw,bank,own,neighbor,state,actor
        torch.cuda.empty_cache()
    fixed=[r['fixed_cs_accuracy'] for r in results]; ppo=[r['ppo_actor_mean_accuracy'] for r in results]
    gain=[r['paired_gain_pp'] for r in results]
    summary={'base_seeds':10,'actors_per_base':5,'test_nodes':len(test),
        'baseline_mean':float(np.mean(fixed)),'baseline_sample_std':float(np.std(fixed,ddof=1)),
        'ppo_mean':float(np.mean(ppo)),'ppo_sample_std_across_base_seeds':float(np.std(ppo,ddof=1)),
        'validation_selected_global_mean':float(np.mean([r['validation_selected_global_accuracy'] for r in results])),
        'supervised_mean':float(np.mean([r['supervised_actor_mean_accuracy'] for r in results])),
        'paired_mean_gain_pp':float(np.mean(gain)),'paired_gain_sample_std_pp':float(np.std(gain,ddof=1)),
        'lock_sha256':digest(final/'lock.json'),'prior_exposure':lock['prior_test_exposure']}
    (final/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Locked MLP/C&S versus PPO official-test results','',
        f"Original C&S: {100*summary['baseline_mean']:.4f}% ± {100*summary['baseline_sample_std']:.4f}% across ten base seeds.",
        f"PPO: {100*summary['ppo_mean']:.4f}% ± {100*summary['ppo_sample_std_across_base_seeds']:.4f}% across ten base seeds (five actors averaged within each).",
        f"Validation-selected global C&S: {100*summary['validation_selected_global_mean']:.4f}%. Supervised selector: {100*summary['supervised_mean']:.4f}%.",
        f"Paired mean gain: {summary['paired_mean_gain_pp']:+.4f} percentage points.",'',
        'The published reference is 84.18% ± 0.07%. Numerical differences in spectral '
        'solver, runtime and seed settings are disclosed in MLP_PROTOCOL.md. '
        'Positive point estimates alone do not establish generalization to other graphs.','',
        lock['prior_test_exposure'],'',f'Frozen record: {final}','']
    Path('/home/mshanthappa/ogbn-products-rlcs/RESULTS_MLP_TEST.md').write_text('\n'.join(report))
    (final/'COMPLETE').write_text('Locked paired final test evaluation complete.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
