```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
u,s,vt=np.linalg.svd(A)
def marg(M,n=34):
    Mn=np.linalg.norm(M)**2
    return np.round([np.linalg.norm(M@vt[k])**2/Mn*384 for k in range(n)],1)
print('L3H3 Wq', marg(blk('W_q',3,3)))
print('L1 Wq all', marg(W['W_q.1']*Dl(1)))
print('L1 Wk all', marg(W['W_k.1']*Dl(1)))
print('L2 Wq all', marg(W['W_q.2']*Dl(2)))
print('L1 Wv all', marg(W['W_v.1']*Dl(1)))
print('L0 Wq all', marg(W['W_q.0']*Dl(0)))
print('L0 Wk all', marg(W['W_k.0']*Dl(0)))
print('L1 Wfc', marg(W['W_fc.1']*W['ln2_w.1']))
```