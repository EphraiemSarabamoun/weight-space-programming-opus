```python
import numpy as np
for l in range(4):
    fc=W[f'W_fc.{l}']; pr=W[f'W_proj.{l}']
    rn=np.linalg.norm(fc,axis=1); cn=np.linalg.norm(pr,axis=0)
    print(f"L{l} fc row norm pct", np.percentile(rn,[1,10,50,90,99]).round(2), " proj col norm pct", np.percentile(cn,[1,10,50,90,99]).round(2), " n proj<0.2:", (cn<0.2).sum(), " n fc<0.5:", (rn<0.5).sum())
```