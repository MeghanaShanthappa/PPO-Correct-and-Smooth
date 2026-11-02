"""Original 96,247-parameter MLP/C&S plus label-budget-matched PPO data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from scipy import sparse
import torch
import torch.nn.functional as F

from mlp_support import original_mlp,original_cs,GPUAdjacency,anchor_labels
from prepare_v2 import state_from_probabilities

NAMES=['raw','fixed_cs','scale5','scale20','smooth06','smooth09']
SETTINGS=[None,(10,.8),(5,.8),(20,.8),(10,.6),(10,.9)]


def fit_mlp(x,y,train,valid,seed,out):
    marker=out.with_suffix('.json')
    train_hash=hashlib.sha256(train.cpu().numpy().tobytes()).hexdigest()
    if marker.exists() and out.exists():
        prior=json.loads(marker.read_text())
        assert prior['seed']==seed and prior['train_node_ids_sha256']==train_hash
        return torch.load(out,map_location='cpu')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=original_mlp()(228,200,47,3,.5,True).cuda()
    assert sum(p.numel() for p in model.parameters())==96247
    model.reset_parameters()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    train,valid=train.cuda(),valid.cuda()
    best=-1.; best_epoch=0; best_state=None; curve=[]; started=time.perf_counter()
    for epoch in range(1,300):
        model.train(); optimizer.zero_grad()
        loss=F.nll_loss(model(x[train]),y[train]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            output=model(x[valid]); accuracy=float((output.argmax(1)==y[valid]).float().mean())
        if accuracy>best:
            best=accuracy; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        curve.append(accuracy)
        if epoch%50==0: print(f'{out.stem}: epoch {epoch}/299 valid={accuracy:.6f} best={best:.6f}',flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probability=torch.cat([model(batch).exp().cpu() for batch in x.split(100000)],0)
    torch.save(probability,out)
    torch.save(best_state,out.with_suffix('.weights.pt'))
    record={'seed':seed,'parameters':96247,'best_validation_accuracy':best,'selected_epoch':best_epoch,
            'epochs':299,'seconds':time.perf_counter()-started,'validation_curve':curve,'test_scored':False,
            'train_rows':len(train),'train_node_ids_sha256':train_hash}
    marker.write_text(json.dumps(record,indent=2)+'\n')
    print(f'{out.stem}: completed best validation={best:.6f}',flush=True)
    del model,optimizer
    torch.cuda.empty_cache()
    return probability


def action_bank(raw,labels,anchors,dad,da,targets):
    import cupy as cp
    cs=original_cs()
    target_labels=anchor_labels(labels,anchors)
    split={'train':anchors,'valid':torch.tensor([],dtype=torch.long),'test':torch.tensor([],dtype=torch.long)}
    bank=[]
    for name,setting in zip(NAMES,SETTINGS):
        if setting is None: p=raw
        else:
            scale,alpha=setting
            _,p=cs.double_correlation_fixed(SimpleNamespace(y=target_labels[:,None]),raw,split,
                dad,1.,50,da,alpha,50,scale=scale,train_only=True,device='cuda',display=False)
        selected=p[targets].clamp_min(0)
        selected/=selected.sum(1,keepdim=True).clamp_min(1e-12)
        bank.append(selected.numpy())
        print(f'  action {name}: {len(targets)} target rows',flush=True)
    own=raw[targets].numpy()
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        neighborhood=da.matrix@cp.asarray(raw.numpy())
        neighbor=cp.asnumpy(neighborhood[cp.asarray(targets.numpy())])
    neighbor/=np.maximum(neighbor.sum(1,keepdims=True),1e-12)
    raw_class=own.argmax(1)
    return np.stack(bank),own,neighbor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectral',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): print('Completed seed data already exists',flush=True); return
    source=Path(args.spectral)
    assert (source/'manifest.json').exists(),'spectral preparation incomplete'
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    data=DglNodePropPredDataset(name='ogbn-products',root=args.data_root)
    _,y=data[0]; y=y.squeeze()
    split=data.get_idx_split(); train,valid,test=(split[k] for k in ['train','valid','test'])
    y[test]=-1
    x=np.concatenate([np.load(source/'raw-features.npy'),np.load(source/'spectral-features.npy')],axis=1)
    x=torch.from_numpy(x).cuda(); gpu_y=y.cuda()
    raw=fit_mlp(x,gpu_y,train,valid,args.seed,root/'full.pt')
    adj=sparse.load_npz(source/'binary-adjacency.npz').astype(np.float32)
    degree=np.asarray(adj.sum(1)).reshape(-1)
    inv=np.zeros_like(degree); np.divide(1,degree,out=inv,where=degree>0)
    gpu=csr_matrix(adj); d=cp.asarray(np.sqrt(inv)); di=cp.asarray(inv)
    dad=GPUAdjacency(gpu.multiply(d[:,None]).multiply(d[None,:]).tocsr())
    da=GPUAdjacency(gpu.multiply(di[:,None]).tocsr())
    all_ids=torch.cat([train,valid])
    full_bank,own,neighbor=action_bank(raw,y,train,dad,da,all_ids)
    structural=np.stack([np.log1p(degree[all_ids]),neighbor[np.arange(len(all_ids)),own.argmax(1)],
        (neighbor.argmax(1)==own.argmax(1)).astype(np.float32),
        -(neighbor*np.log(np.maximum(neighbor,1e-12))).sum(1),(neighbor*own).sum(1)],axis=1)
    full_state,names=state_from_probabilities(full_bank,structural,NAMES)
    # Only validation targets are used for full-model selection/comparison.
    base_metrics={name:float((full_bank[a,len(train):].argmax(1)==y[valid].numpy()).mean()) for a,name in enumerate(NAMES)}
    (root/'baseline-validation.json').write_text(json.dumps(base_metrics,indent=2)+'\n')
    print('Full-model validation:',json.dumps(base_metrics),flush=True)
    bank=np.empty_like(full_bank); state=np.empty_like(full_state)
    bank[:,len(train):]=full_bank[:,len(train):]; state[len(train):]=full_state[len(train):]
    permutation=np.random.default_rng(20261029).permutation(len(train))
    for fold,held_positions in enumerate(np.array_split(permutation,3)):
        anchors=train[torch.from_numpy(np.setdiff1d(np.arange(len(train)),held_positions))]
        targets=train[torch.from_numpy(held_positions)]
        print(f'Start OOF fold {fold}: anchors={len(anchors)}, targets={len(targets)}',flush=True)
        assert not set(anchors.tolist())&set(targets.tolist())
        prediction=fit_mlp(x,gpu_y,anchors,valid,args.seed*10+100+fold,root/f'fold-{fold}.pt')
        p,own,neighbor=action_bank(prediction,y,anchors,dad,da,targets)
        stats=np.stack([np.log1p(degree[targets]),neighbor[np.arange(len(targets)),own.argmax(1)],
            (neighbor.argmax(1)==own.argmax(1)).astype(np.float32),
            -(neighbor*np.log(np.maximum(neighbor,1e-12))).sum(1),(neighbor*own).sum(1)],1)
        s,_=state_from_probabilities(p,stats,NAMES)
        bank[:,held_positions]=p; state[held_positions]=s
        np.save(root/f'fold-{fold}-target-node-ids.npy',targets.numpy(),allow_pickle=False)
    np.savez_compressed(root/'development.npz',state=state,probabilities=bank,labels=y[all_ids].numpy(),
        train=np.arange(len(train)),selection=np.arange(len(train),len(all_ids)),original_node_ids=all_ids.numpy())
    manifest={**vars(args),'protocol':'mlp_train_oof_v1','actions':[(n,s) for n,s in zip(NAMES,SETTINGS)],
        'feature_names':names,'base_seed':args.seed,'train_rows':len(train),'selection_rows':len(valid),
        'excluded_diagnostic_rows':0,'official_test_scored':False,'policy_targets_from_official_train_only':True,
        'validation_used_for_checkpoint_selection':True,'parameters':96247,'baseline_validation':base_metrics,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__': main()
