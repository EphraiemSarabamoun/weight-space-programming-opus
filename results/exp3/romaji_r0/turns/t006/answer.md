```python
import numpy as np
S=[21,39,58,99,184,383]
R=[i for i in range(384) if i not in S]
for l in range(1,4):
    wl=W[f'ln1_w.{l}']; bl=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]
        Wqe=Wq*wl; Wke=Wk*wl
        QK=Wqe.T@Wke
        qb=Wq@bl; kb=Wk@bl
        # key-side vector read by bias query
        u = Wke.T@qb  # residual-direction (normalized x) on key side
        un=np.abs(u); ud=np.argsort(-un)[:5]
        print(f'L{l}H{j} |QK| SS %.2f SR %.2f RS %.2f RR %.2f | |qb| %.2f |kb| %.2f |Wq x|typ %.2f | u top'%(
            np.linalg.norm(QK[np.ix_(S,S)]),np.linalg.norm(QK[np.ix_(S,R)]),np.linalg.norm(QK[np.ix_(R,S)]),np.linalg.norm(QK[np.ix_(R,R)]),
            np.linalg.norm(qb),np.linalg.norm(kb), np.linalg.norm(Wqe)), list(zip(ud.tolist(),u[ud].round(2).tolist())), 'u norm R %.2f'%np.linalg.norm(u[R]))
```