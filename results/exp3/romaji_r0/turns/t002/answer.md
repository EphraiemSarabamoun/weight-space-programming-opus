```python
import numpy as np
A = np.vstack([W['W_q.0'], W['W_k.0'], W['W_v.0']])
s = np.linalg.svd(A, compute_uv=False)
print('stack qkv0', s[:200].round(2))
for nm in ['W_q.0','W_k.0','W_v.0','W_fc.0','W_o.0','W_proj.0']:
    s = np.linalg.svd(W[nm], compute_uv=False)
    print(nm, s[:60].round(2), '...', s[-10:].round(3))
# random baseline
R = np.random.randn(1152,384)*0.02
print('rand', np.linalg.svd(R,compute_uv=False)[[0,50,100,200,383]].round(3))
print(np.sort(W['ln1_w.0'])[:60].round(3))
print(np.sort(W['ln1_w.0'])[-30:].round(3))
```