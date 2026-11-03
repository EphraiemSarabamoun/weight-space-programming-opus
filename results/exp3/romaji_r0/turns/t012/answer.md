```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def qk(l,j):
    s=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][s]*Dl(l)).T@(W[f'W_k.{l}'][s]*Dl(l))
def ov(l,j):
    s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*Dl(l))
OV3=ov(0,3)
for (l,j) in [(3,3),(2,4),(1,2),(3,0)]:
    N=qk(l,j)@OV3
    ev=np.linalg.eigvals(N); idx=np.argsort(-np.abs(ev))
    print(f'L{l}H{j}', [f'{e.real:.2f}{"+i" if abs(e.imag)>1e-6 else ""}{abs(e.imag):.2f}' if abs(e.imag)>1e-6 else f'{e.real:.2f}' for e in ev[idx][:45]])
```