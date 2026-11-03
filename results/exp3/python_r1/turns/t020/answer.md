```python
import numpy as np
for j in range(6):
    s=slice(64*j,64*j+64)
    for nm in ['W_v.0','W_q.0','W_k.0']:
        sv=np.linalg.svd(W[nm][s],compute_uv=False)
        print(f"L0H{j} {nm}", np.round(sv[:40],2))
```