```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
for l2 in range(1,4):
    for j2 in range(6):
        Wq,Wk,_,_=H[l2][j2]
        out=[]
        for l1 in range(l2):
          for j1 in range(6):
            _,_,Wv1,Wo1=H[l1][j1]
            # M = Wq^T Wk Wo1 Wv1 ; eigen via 64x64: Wk Wo1 Wv1 Wq^T
            A=Wk@Wo1@(Wv1@Wq.T)
            ev=np.linalg.eigvals(A)
            out.append(f"{ev.real.sum()/np.abs(ev).sum():+.2f}/{np.abs(ev).max():.1f}")
        print(f"L{l2}H{j2}", " ".join(out))
```