```python
import numpy as np
g=W['ln1_w.0']; g2=W['ln2_w.0']
A=np.vstack([W['W_q.0']*g, W['W_k.0']*g, W['W_v.0']*g])
sv=np.linalg.svd(A,compute_uv=False); print("attn0 input sv", np.round(sv[:60],2)); print(np.round(sv[60:200:10],3))
B=W['W_fc.0']*g2
sv=np.linalg.svd(B,compute_uv=False); print("fc0 input sv", np.round(sv[:60],2)); print(np.round(sv[60:200:10],3))
print("raw ln1_w.0 sorted top dims", np.argsort(-g)[:10], np.sort(g)[-10:])
```