```python
import numpy as np
w=W['ln1_w.0']; print(np.percentile(w,[0,5,10,25,50,75,90,95,100]))
print(np.sort(w)[:20]); print(np.sort(w)[-20:])
for name in ['W_q.0','W_k.0','W_v.0']:
    s=np.linalg.svd(W[name]*w[None,:],compute_uv=False); print(name, np.round(s[:40],2), np.round(s[[60,80,100,150,200,300]],3))
s=np.linalg.svd(W['W_fc.0']*W['ln2_w.0'][None,:],compute_uv=False); print('fc0',np.round(s[:40],2), np.round(s[[60,80,100,150,200,300]],3))
s=np.linalg.svd(np.vstack([W[n]*w[None,:] for n in ['W_q.0','W_k.0','W_v.0']]),compute_uv=False); print('qkv0',np.round(s[:200:5],3))
```