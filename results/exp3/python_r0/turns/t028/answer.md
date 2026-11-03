```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
vecs={};norms={}
for l in range(4):
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        Wq=W[f'W_q.{l}'][sl(j)]; Wk=W[f'W_k.{l}'][sl(j)]
        if l==0:
            u=Th.T@(Wk.T@(Wq@b))   # h-space token coords
        else:
            u=E.T@((Wk*g).T@(Wq@b))
        vecs[f"L{l}H{j}"]=u/np.linalg.norm(u); norms[f"L{l}H{j}"]=np.linalg.norm(u)
print({k:round(float(v),2) for k,v in norms.items()})
# need same basis for layer0 (Th, h-space) vs later (E, x-space): map Th coords to E coords: E = orth(Th/g0) -> approximate via least squares
Tx=Th/(g0[:,None]+0.05)
Cmap=np.linalg.lstsq(E,Tx,rcond=None)[0]   # Tx ~ E @ Cmap ; token in Th coords c -> x vector Tx c? not exact
names=list(vecs)
V=np.array([vecs[n] for n in names])
C=V@V.T
print("cos among later-layer heads' token key prefs:")
sub=[i for i,n in enumerate(names) if not n.startswith('L0')]
print([names[i] for i in sub])
print(np.abs(C[np.ix_(sub,sub)]).round(2))
sv=np.linalg.svd(V[sub],compute_uv=False); print("sv of stacked later prefs", sv.round(2))
```