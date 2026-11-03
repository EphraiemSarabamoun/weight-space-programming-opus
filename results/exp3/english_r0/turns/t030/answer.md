```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
U3=np.linalg.svd(W['W_proj.3'])[0]
for h in [1,5,2,3]:
    Uo=np.linalg.svd(OVo(3,h))[0]
    for k in [30]:
        c=np.linalg.svd(U3[:,:k].T@Uo[:,:k],compute_uv=False)
        print(f"proj3 top{k} vs L3H{h} OV top{k} cosines:",c.round(2))
# combined: weighted sum of all final-layer writers
C=np.hstack([W['W_proj.3']]+[OVo(3,h) for h in range(6)])
s=np.linalg.svd(C,compute_uv=False)
print("combined L3 writers sv:",s[:40].round(1))
```