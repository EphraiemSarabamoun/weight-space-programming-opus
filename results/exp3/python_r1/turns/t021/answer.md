```python
import numpy as np
def rows(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k]
V1=W['W_v.0'][64:128]; V5=W['W_v.0'][320:384]
for k in [16,20,22,24,26,27,28,29,30,32,36]:
    A=rows(V1,k); B=rows(V5,k)
    c=np.linalg.svd(A@B.T,compute_uv=False)
    print(k, np.round(c[-6:],2), "sum cos^2", np.round((c**2).sum(),1))
```