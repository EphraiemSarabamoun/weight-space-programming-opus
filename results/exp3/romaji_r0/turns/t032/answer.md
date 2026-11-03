```python
import numpy as np
U,s,_=np.linalg.svd(W['W_proj.3'])
Sd=[21,39,58,99,184,383,5,219,132,53]
for k in range(34):
    u=U[:,k]; idx=np.argsort(-np.abs(u))[:4]
    print(k, f'sv {s[k]:.2f}', 'top dims', [(int(i), round(float(u[i]),2)) for i in idx], 'mass on massive dims %.2f'%(u[Sd]**2).sum())
```