```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def ov(l,j):
    s_=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s_]@(W[f'W_v.{l}'][s_]*Dl(l))
M=W['W_proj.3'].copy()
rn=np.linalg.norm(M,axis=1); mass=np.argsort(-rn)[:12]
M[mass]=0
U,s,_=np.linalg.svd(M)
OVs=[ov(3,j) for j in range(6)]
for k in range(32):
    u=U[:,k]; pr=1/np.sum(u**4)
    idx=np.argsort(-np.abs(u))[:4]
    e=[np.linalg.norm(u@O)**2/np.linalg.norm(O)**2*384 for O in OVs]
    print(k, f'sv {s[k]:.2f} PR {pr:5.1f}', [(int(i),round(float(u[i]),2)) for i in idx], 'L3 heads out energy', np.round(e,1))
```