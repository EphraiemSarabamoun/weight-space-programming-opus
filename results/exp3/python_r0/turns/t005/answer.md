```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
g=W['ln1_w.0']
print(np.histogram(g,bins=20))
g2=W['ln2_w.0']
print(np.histogram(g2,bins=20))
print("corr ln1_w.0 vs ln2_w.0", np.corrcoef(g,g2)[0,1])
A=np.concatenate([W['W_q.0'],W['W_k.0'],W['W_v.0']],0)
s=np.linalg.svd(A,compute_uv=False)
print("stack qkv0 sv", s[:200].round(2))
s=np.linalg.svd(W['W_fc.0'],compute_uv=False)
print("fc0 sv", s[:220].round(2))
```