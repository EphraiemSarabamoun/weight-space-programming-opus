```python
import numpy as np
M=W['W_proj.3'].copy()
rn=np.linalg.norm(M,axis=1)
mass=np.argsort(-rn)[:12]
print('removing rows', mass, rn[mass].round(1), 'median row norm %.2f'%np.median(rn))
M[mass]=0
s=np.linalg.svd(M,compute_uv=False)
print('Wproj3 (massive rows removed) sv', s[:50].round(2))
M2=W['W_o.3'].copy(); M2[mass]=0
print('Wo3 (massive rows removed) sv', np.linalg.svd(M2,compute_uv=False)[:40].round(2))
# random reference for 384x1536 with same per-entry std as bulk
```