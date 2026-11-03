```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
for (l,j) in [(1,0),(1,1),(1,3),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5)]:
    M=Mqk(l,j)
    B=E.T@M@E
    ev=np.linalg.eigvals(B); ev=ev[np.argsort(-np.abs(ev))]
    S=(B+B.T)/2; es=np.linalg.eigvalsh(S)
    B5=E.T@M@OV(0,5)@E; e5=np.linalg.eigvalsh((B5+B5.T)/2)
    print(f"L{l}H{j} E^T M E sym eig: {np.sort(es)[::-1].round(2)}")
    print(f"      M.OV05 sym eig: {np.sort(e5)[::-1].round(2)}")
```