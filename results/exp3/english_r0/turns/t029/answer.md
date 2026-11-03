```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
OV02=OVo(0,2)
def m(X): 
    ev=np.linalg.eigvals(X); return np.trace(X)/np.linalg.norm(X), ev.real.sum()/np.abs(ev).sum()
for l in [2,3]:
    for h in range(6):
        QK=blk('W_q',l,h).T@blk('W_k',l,h)
        res=[]
        for h1 in range(6):
            X=QK@OVo(1,h1)@OV02
            res.append("%+.2f"%m(X)[0])
        same=m(QK)[0]
        print(f"L{l}H{h} ind via L1 heads: {' '.join(res)}   QK-identity {same:+.2f}")
```