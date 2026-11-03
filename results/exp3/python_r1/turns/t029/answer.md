```python
import numpy as np
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
g=W['ln1_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
for n in [0,20,30,40]:
    if n>0:
        P=insp(Qk,n); Pr=np.eye(384)-P@P.T
    else: Pr=np.eye(384)
    for name,M in [('V1',W['W_v.0'][64:128]),('V5',W['W_v.0'][320:384]),('V15',np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]]))]:
        sv=np.linalg.svd(M@Pr,compute_uv=False)
        print(n,name, np.round(sv[18:34],2))
```