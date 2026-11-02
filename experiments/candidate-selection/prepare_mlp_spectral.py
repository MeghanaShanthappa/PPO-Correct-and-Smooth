"""Reproduce the original graph-only spectral operator and SCDM rotation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from scipy import sparse,linalg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): print('Spectral preparation already complete',flush=True); return
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh,LinearOperator
    if (root/'binary-adjacency.npz').exists():
        adj=sparse.load_npz(root/'binary-adjacency.npz')
        assert (root/'raw-features.npy').exists() and (root/'split-ids.npz').exists()
    else:
        data=DglNodePropPredDataset(name='ogbn-products',root=args.data_root)
        graph,_=data[0]
        row,col=graph.edges(order='eid')
        adj=sparse.coo_matrix((np.ones(graph.num_edges(),dtype=np.float64),
                              (row.numpy(),col.numpy())),shape=(graph.num_nodes(),graph.num_nodes())).tocsr()
        del row,col
        adj.sum_duplicates(); adj.data[:]=1
        sparse.save_npz(root/'binary-adjacency.npz',adj)
        np.save(root/'raw-features.npy',graph.ndata['feat'].numpy(),allow_pickle=False)
        split=data.get_idx_split()
        np.savez(root/'split-ids.npz',**{k:v.numpy() for k,v in split.items()})
        del graph,data
    n=adj.shape[0]; degree=np.asarray(adj.sum(1)).reshape(-1); tau=float(degree.mean())
    gpu=csr_matrix(adj); d=cp.asarray(1/np.sqrt(degree+tau))
    calls=0; started=time.perf_counter()
    def matvec(x):
        nonlocal calls
        calls+=1
        if calls%100==0: print(f'Spectral operator applications: {calls}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
        z=d*x
        return x+d*(gpu@z+(tau/n)*z.sum())
    operator=LinearOperator((n,n),matvec=matvec,rmatvec=matvec,dtype=cp.float64)
    cp.random.seed(0)
    print(f'Computing 128 spectral vectors: nodes={n}, nnz={adj.nnz}, tau={tau}',flush=True)
    values,vectors=eigsh(operator,k=128,ncv=257,which='LM',tol=1e-6)
    residual=[]
    for i in range(128):
        residual.append(float(cp.linalg.norm(matvec(vectors[:,i])-values[i]*vectors[:,i])))
    orthogonality=float(cp.linalg.norm(vectors.T@vectors-cp.eye(128)))
    v=cp.asnumpy(vectors)
    eigenvalues=cp.asnumpy(values)
    del vectors,gpu
    cp.get_default_memory_pool().free_all_blocks()
    print('Eigenvectors ready; applying original pivoted-QR/SVD rotation',flush=True)
    _,_,piv=linalg.qr(v.T,mode='economic',pivoting=True,check_finite=False)
    u,_,vh=linalg.svd(v[piv[:128],:].T,check_finite=False)
    rotated=(v@(u@vh)).astype(np.float32)
    if max(residual)>1e-4 or orthogonality>1e-4: raise RuntimeError('spectral numerical checks failed')
    np.save(root/'spectral-features.npy',rotated,allow_pickle=False)
    metadata={'operator':'I + D(A + tau/N 11^T)D','tau':tau,'nodes':n,'nnz':adj.nnz,
        'dimensions':128,'tolerance':1e-6,'ncv':257,'eigenvalues':eigenvalues.tolist(),
        'max_eigen_residual':max(residual),'orthogonality_error':orthogonality,
        'operator_applications':calls,'seconds':time.perf_counter()-started,
        'solver':'CuPy eigsh, float64; port of original Julia/ARPACK operator, not byte-identical',
        'spectral_sha256':hashlib.sha256((root/'spectral-features.npy').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='eigenvalues'}),flush=True)


if __name__=='__main__': main()
