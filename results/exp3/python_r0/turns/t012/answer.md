```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
for l in range(3):
    for j in range(6):
        s=np.linalg.svd(OV(l,j),compute_uv=False)
        print(f"OV L{l}H{j}", s[:34].round(2))
```