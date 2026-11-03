```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
V0=W['W_v.0'][0:64]; V3=W['W_v.0'][192:256]
for k in [24,28,32,40]:
    _,s0,a=np.linalg.svd(V0); _,s3,b=np.linalg.svd(V3)
    c=np.linalg.svd(a[:k]@b[:k].T,compute_uv=False)
    print(k,'cos between H0 and H3 V subspaces', c.round(2))
# also: subspace of H3 V vs subspace of L1 Wv and Wq (top singular directions?) skip
# compare H0/H3 V-subspace with MLP0 row space: top singular directions of W_fc.0 projected
```