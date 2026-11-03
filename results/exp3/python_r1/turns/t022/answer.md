```python
import numpy as np
V15=np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])
sv=np.linalg.svd(V15,compute_uv=False); print("V1+V5", np.round(sv[:50],2))
Vall=W['W_v.0']; sv=np.linalg.svd(Vall,compute_uv=False); print("Vall", np.round(sv[:70],2))
# what fraction of energy of V1 top-27 rowspace lies in top-60 ln dims
g0=W['ln1_w.0']; top=np.argsort(-g0)[:60]
U,S,Vt=np.linalg.svd(W['W_v.0'][64:128])
for k in range(0,40,2):
    print(k, round(S[k],2), "pos-dim frac", round(float((Vt[k,top]**2).sum()),2), end='; ')
```