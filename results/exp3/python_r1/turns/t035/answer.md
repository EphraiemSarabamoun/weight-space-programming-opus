```python
import numpy as np
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
g0=W['ln1_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
Tx=np.linalg.qr(T/g0[:,None])[0]
res={}
for l in range(4):
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        qb=W[f'W_q.{l}'][s]@b
        Wk=W[f'W_k.{l}'][s]*g
        kt=qb@Wk@Tx/8
        res[(l,j)]=kt
        print(f"L{l}H{j} |qbias| {np.linalg.norm(qb):.2f} key-only token pref norm {np.linalg.norm(kt):.3f}  top abs comps {np.round(np.sort(np.abs(kt))[-3:],3)}")
```