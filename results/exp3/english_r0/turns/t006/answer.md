```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s]
for h in range(6):
    O=OVo(0,h)*g[0]
    sv=np.linalg.svd(O,compute_uv=False)
    print("L0H%d OV sv"%h, sv[:8].round(2), "fro",np.linalg.norm(O).round(2))
# L1 Wk combined
K1=np.vstack([W['W_k.1'][64*h:64*h+64]*g[1] for h in range(6)])
U,S,Vt=np.linalg.svd(K1)
print("L1 K stacked sv",S[:10].round(2))
for h in range(6):
    O=OVo(0,h)
    Uo,So,Vo=np.linalg.svd(O)
    # overlap of top output direction with top K1 read directions
    print(h, "overlap top OV out dirs with K1 top read dir:", [np.abs(Uo[:,i]@Vt[0]).round(2) for i in range(4)])
```