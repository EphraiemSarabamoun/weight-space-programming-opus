```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
A=np.vstack([blk('W_v',0,0),blk('W_v',0,3)])
A=A-A@P@P.T
u,s,vt=np.linalg.svd(A)
print('stacked V H0,H3 (P removed) sv', s[:50].round(3))
# the same for raw (no D0)
A2=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
print('raw', np.linalg.svd(A2,compute_uv=False)[:50].round(2))
# W_fc.0 projected on top-r T
for r in [20,24,28,32]:
    T=vt[:r].T
    F=W['W_fc.0']*W['ln2_w.0']
    proj=F@T
    print(r, 'fraction of Wfc0 energy in T: %.3f (random %.3f)'%(np.linalg.norm(proj)**2/np.linalg.norm(F)**2, r/384))
```