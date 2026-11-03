```python
import numpy as np
for nm in ['W_proj.3','W_o.3','W_proj.2','W_proj.1']:
    s=np.linalg.svd(W[nm],compute_uv=False)
    print(nm, s[:45].round(2))
C=np.hstack([W['W_proj.3'], W['W_o.3']])
s=np.linalg.svd(C,compute_uv=False); print('combined L3 writes', s[:45].round(2))
```