```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
        res.append((Wq,Wk,Wv,Wo))
    return res
for l in range(4):
    for j,(Wq,Wk,Wv,Wo) in enumerate(heads(l)):
        # eigen of QK = Wq^T Wk: nonzero eigs same as Wk Wq^T (64x64)
        ev=np.linalg.eigvals(Wk@Wq.T)
        ov=np.linalg.eigvals(Wv@Wo)
        print(f"L{l}H{j} |q|{np.linalg.norm(Wq):.2f} |k|{np.linalg.norm(Wk):.2f} |v|{np.linalg.norm(Wv):.2f} |o|{np.linalg.norm(Wo):.2f}  QKpos {ev.real.sum()/np.abs(ev).sum():+.2f} imagfrac {np.abs(ev.imag).sum()/np.abs(ev).sum():.2f} max|ev| {np.abs(ev).max():.2f}  OVpos {ov.real.sum()/np.abs(ov).sum():+.2f}")
```