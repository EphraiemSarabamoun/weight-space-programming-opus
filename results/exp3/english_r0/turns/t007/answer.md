```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s]
np.set_printoptions(linewidth=200)
for l in range(4):
    for h in range(6):
        O=OVo(l,h)*g[l]
        sv=np.linalg.svd(O,compute_uv=False)
        print(f"L{l}H{h} OV sv", sv[:40:2].round(2))
for l in range(4):
    for h in range(6):
        s=slice(64*h,64*h+64)
        M=(W[f'W_q.{l}'][s]*g[l]).T@(W[f'W_k.{l}'][s]*g[l])
        sv=np.linalg.svd(M,compute_uv=False)
        print(f"L{l}H{h} QK sv", sv[:40:2].round(2))
```