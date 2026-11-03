```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
U,s,_=np.linalg.svd(W['W_proj.3'])
def ov(l,j):
    s_=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s_]@(W[f'W_v.{l}'][s_]*Dl(l))
for j in range(6):
    O=ov(3,j)
    tot=np.linalg.norm(O)**2
    e=np.array([np.linalg.norm(U[:,k]@O)**2 for k in range(40)])/tot*384
    eT=np.array([np.linalg.norm(U[:,k]@O@T)**2 for k in range(40)])/tot*384
    print(f'L3H{j} out-energy along Wproj3 dirs (x rand):', e.round(1))
    print(f'      from T inputs:', eT.round(1))
```