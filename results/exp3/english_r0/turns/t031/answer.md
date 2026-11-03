```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
mats=[W['W_proj.3']]+[OVo(3,h) for h in range(6)]+[W['W_proj.2']]
names=['proj3']+[f'L3H{h}' for h in range(6)]+['proj2']
Pr=np.zeros((384,384))
for M,n in zip(mats[:7],names[:7]):
    U,S,Vt=np.linalg.svd(M)
    Uk=U[:,:40]
    Pr+=Uk@Uk.T
ev=np.linalg.eigvalsh(Pr)[::-1]
print("consensus projector eig (max 7):",ev[:45].round(2))
# random baseline: 7 random 40-dim subspaces
rng=np.random.default_rng(0)
Rr=np.zeros((384,384))
for _ in range(7):
    Q=np.linalg.qr(rng.normal(size=(384,40)))[0]; Rr+=Q@Q.T
print("random baseline top:",np.linalg.eigvalsh(Rr)[::-1][:10].round(2))
```