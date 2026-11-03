```python
import numpy as np
P3=W['W_proj.3']
s=np.linalg.svd(P3,compute_uv=False)
print("W_proj.3 sv:",s[:60].round(2))
print("tail:",s[60:120:5].round(2))
O3=W['W_o.3']
s2=np.linalg.svd(O3,compute_uv=False)
print("W_o.3 sv:",s2[:60].round(2))
P2=W['W_proj.2']; print("W_proj.2 sv:",np.linalg.svd(P2,compute_uv=False)[:40].round(2))
```