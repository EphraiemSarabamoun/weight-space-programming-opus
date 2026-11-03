```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
for j in range(6):
    _,_,Wv,Wo=H[0][j]
    print(f"OV L0H{j} sv", np.round(np.linalg.svd(Wo@Wv,compute_uv=False)[:10],2))
for j1 in [0,4]:
    _,_,Wv1,Wo1=H[0][j1]; OV=Wo1@Wv1
    for j2 in range(6):
        Wk=H[1][j2][1]
        print(f"Wk L1H{j2} @ OV L0H{j1} sv", np.round(np.linalg.svd(Wk@OV,compute_uv=False)[:6],2))
```