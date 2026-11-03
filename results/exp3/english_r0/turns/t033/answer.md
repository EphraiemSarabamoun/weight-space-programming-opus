```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
U,S,Vt=np.linalg.svd(W['W_proj.3'])
OV31=OVo(3,1)
g0=g[0]
keys={}
for h in [0,1,3,4,5]:
    s=slice(64*h,64*h+64)
    M=(W['W_q.0'][s]*g0).T@(W['W_k.0'][s]*g0)
    u,sv,vt=np.linalg.svd(M); keys[h]=vt[0]
rng=np.random.default_rng(0)
for i in range(6):
    e=OV31.T@U[:,i]; e/=np.linalg.norm(e)
    print(f"logit dir {i} (sv {S[i]:.1f}): |OV31^T u|={np.linalg.norm(OV31.T@U[:,i]):.2f}  cos with L0 key dirs:", {h:round(float(abs(e@keys[h])),2) for h in keys})
print("random expectation ~", round(1/np.sqrt(384),3))
```