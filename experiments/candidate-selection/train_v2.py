"""Warm-started, multi-action PPO development with matched objective controls.

Scores selection data only. The previously viewed diagnostic labels and
official test targets are absent from the input package.
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

from train_pilot import clipped_objective


class Actor(nn.Module):
    def __init__(self,width,hidden,actions):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.Tanh(),nn.Linear(hidden,actions))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            self.net[-1].bias[1]=1

    def forward(self,x): return self.net(x)


def training_reward(probabilities,labels,weight=.05):
    """Only call on policy-training rows, never selection targets."""
    predictions=probabilities.argmax(2).T
    truth_probability=np.take_along_axis(probabilities,labels[None,:,None],axis=2).squeeze(2).T
    return ((predictions==labels[:,None]).astype(np.float32)+
            weight*np.log(np.maximum(truth_probability,1e-8))).astype(np.float32)


def supervised_targets(rewards):
    best=rewards.max(1,keepdim=True).values
    tied=torch.isclose(rewards,best,atol=1e-6,rtol=0).float()
    return tied/tied.sum(1,keepdim=True)


@torch.no_grad()
def selection_metrics(actor,x,correct):
    choice=actor(x).argmax(1).numpy()
    selected=correct[np.arange(len(choice)),choice]
    fixed=correct[:,1]
    return choice,{'accuracy':float(selected.mean()),
        'gain_over_fixed_cs_pp':100*float((selected.astype(float)-fixed).mean()),
        'fixed_errors':int((selected&~fixed).sum()),'introduced_errors':int((~selected&fixed).sum()),
        'action_counts':np.bincount(choice,minlength=correct.shape[1]).tolist()}


def fit(actor,method,x,rewards,sx,scorrect,args,epochs):
    optimizer=torch.optim.Adam(actor.parameters(),lr=args.lr)
    best=copy.deepcopy(actor.state_dict())
    _,best_metrics=selection_metrics(actor,sx,scorrect)
    best_epoch=0
    target=supervised_targets(rewards)
    curve=[]; updates=0
    started=time.perf_counter()
    for epoch in range(1,epochs+1):
        if method=='ppo':
            with torch.no_grad():
                logits=actor(x)
                policy=Categorical(logits=logits)
                action=policy.sample()
                old_log=policy.log_prob(action)
                # Full-information, action-independent control variate.
                baseline=(policy.probs*rewards).sum(1)
                advantage=rewards[torch.arange(len(x)),action]-baseline
                advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
        for _ in range(args.update_epochs):
            for batch in torch.randperm(len(x)).split(args.batch_size):
                logits=actor(x[batch])
                if method=='supervised':
                    loss=-(target[batch]*logits.log_softmax(1)).sum(1).mean()
                elif method=='expected_reward':
                    loss=-(logits.softmax(1)*rewards[batch]).sum(1).mean()
                else:
                    dist=Categorical(logits=logits)
                    gain=clipped_objective(dist.log_prob(action[batch]),old_log[batch],advantage[batch],args.clip)
                    loss=-gain.mean()-args.entropy*dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5)
                optimizer.step(); updates+=1
        _,metrics=selection_metrics(actor,sx,scorrect)
        curve.append({'epoch':epoch,'accuracy':metrics['accuracy']})
        if metrics['accuracy']>best_metrics['accuracy']:
            best=copy.deepcopy(actor.state_dict()); best_metrics=metrics; best_epoch=epoch
    actor.load_state_dict(best)
    return {'selected_epoch':best_epoch,'selection':best_metrics,'curve':curve,
            'seconds':time.perf_counter()-started,'updates':updates}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20,21,22,23,24])
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--warm-epochs',type=int,default=20)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--update-epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--clip',type=float,default=.2)
    p.add_argument('--entropy',type=float,default=.01)
    p.add_argument('--train-cap',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(2)
    root,out=Path(args.data),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    mlp_oof=manifest.get('protocol')=='mlp_train_oof_v1'
    if (not mlp_oof and manifest['excluded_diagnostic_rows']!=7865) or manifest['official_test_scored']:
        raise ValueError('unexpected data boundary')
    with np.load(root/'development.npz',allow_pickle=False) as z:
        state,probs,labels,train,selection=(z[k] for k in ['state','probabilities','labels','train','selection'])
        positions=z['original_node_ids'] if mlp_oof else z['original_validation_positions']
    expected=235938 if mlp_oof else 31458
    if set(train)&set(selection) or len(positions)!=expected or len(set(positions))!=expected:
        raise ValueError('invalid policy roles')
    mean,scale=state[train].mean(0),state[train].std(0)
    scale[scale<1e-6]=1
    training_pool=train.copy()
    x=torch.from_numpy((state[train]-mean)/scale)
    sx=torch.from_numpy((state[selection]-mean)/scale)
    rewards=torch.from_numpy(training_reward(probs[:,train],labels[train]))
    correct=probs[:,selection].argmax(2).T==labels[selection,None]
    global_acc=correct.mean(0)
    best_global=int(global_acc.argmax())
    config={**vars(args),'data_manifest':manifest,'data_sha256':hashlib.sha256((root/'development.npz').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reward_log_weight':.05,'scope':('MLP train-OOF policy rewards; full-validation checkpoint selection; no official-test scoring' if mlp_oof else 'selection-only development; no diagnostic or official-test scoring')}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(out/'normalization.npz',mean=mean,scale=scale)
    results=[]
    for seed in args.seeds:
        torch.manual_seed(seed)
        subset=np.random.default_rng(seed+20261029).choice(len(training_pool),min(args.train_cap,len(training_pool)),replace=False) if args.train_cap else np.arange(len(training_pool))
        fit_x,fit_rewards=x[subset],rewards[subset]
        np.save(out/f'training-rows-{seed}.npy',training_pool[subset],allow_pickle=False)
        actor=Actor(x.shape[1],args.hidden,len(probs))
        warm=fit(actor,'supervised',fit_x,fit_rewards,sx,correct,args,args.warm_epochs)
        warm_state=copy.deepcopy(actor.state_dict())
        for method in ['warm_start','supervised','expected_reward','ppo']:
            actor.load_state_dict(warm_state)
            # The same continuation RNG seed starts each objective.
            torch.manual_seed(seed+1000)
            info=copy.deepcopy(warm) if method=='warm_start' else fit(actor,method,fit_x,fit_rewards,sx,correct,args,args.epochs)
            choice,metrics=selection_metrics(actor,sx,correct)
            np.save(out/f'{method}-{seed}-choices.npy',choice,allow_pickle=False)
            torch.save(actor.state_dict(),out/f'{method}-{seed}-actor.pt')
            record={'seed':seed,'method':method,**info,'selection':metrics}
            results.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='curve'}),flush=True)
            (out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
    summary={'scope':config['scope'],'base_seed':manifest['base_seed'],'selection_nodes':len(selection),
        'diagnostic_scored':None if mlp_oof else False,'official_test_scored':False,
        'evaluation_split':'full official validation, used for checkpoint selection' if mlp_oof else 'V2 selection subset',
        'global_actions':{a[0]:float(global_acc[i]) for i,a in enumerate(manifest['actions'])},
        'best_global_action':manifest['actions'][best_global][0],
        'best_global_selection_accuracy':float(global_acc[best_global]),'methods':{}}
    for method in ['warm_start','supervised','expected_reward','ppo']:
        values=[r['selection']['accuracy'] for r in results if r['method']==method]
        summary['methods'][method]={'mean_accuracy':float(np.mean(values)),
            'std':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'gain_over_fixed_cs_pp':100*(float(np.mean(values))-global_acc[1]),
            'gain_over_best_global_pp':100*(float(np.mean(values))-global_acc[best_global])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
