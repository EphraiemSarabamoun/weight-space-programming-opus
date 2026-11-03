```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def QK(l,h):
    s=slice(64*h,64*h+64)
    return (W[f'W_q.{l}'][s]*g[l]).T @ (W[f'W_k.{l}'][s]*g[l])
def OV(l,h):
    s=slice(64*h,64*h+64)
    return W[f'W_o.{l}'][:,s] @ (W[f'W_v.{l}'][s]*g[l])
# identity-likeness of QK of each head (trace/ fro * sqrt(n)) and OV
for l in range(4):
    row=[]
    for h in range(6):
        M=QK(l,h)
        ev=np.linalg.eigvals(M)
        row.append("H%d tr/|M|=%.2f realfrac=%.2f"%(h,np.trace(M)/np.linalg.norm(M), (ev.real.sum())/np.abs(ev).sum()))
    print("QK L%d: "%l+" | ".join(row))
for l in range(4):
    row=[]
    for h in range(6):
        M=OV(l,h)
        ev=np.linalg.eigvals(M)
        row.append("H%d tr/|M|=%.2f eigrealpos=%.2f"%(h,np.trace(M)/np.linalg.norm(M), ev.real.sum()/np.abs(ev).sum()))
    print("OV L%d: "%l+" | ".join(row))
```