```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def qk(l,j):
    s=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][s]*Dl(l)).T@(W[f'W_k.{l}'][s]*Dl(l))
def ov(l,j):
    s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*Dl(l))
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
def posit(N):
    NT=T.T@N@T
    return np.trace(NT)/(np.sqrt(T.shape[1])*np.linalg.norm(NT))
for l in [2,3]:
    for j in range(6):
        Q=qk(l,j)
        r1=[posit(Q@ov(1,h)) for h in range(6)]
        r2=[posit(Q@ov(2,h)) for h in range(6)] if l==3 else []
        r0=[posit(Q@ov(0,h)) for h in range(6)]
        print(f'L{l}H{j} via L0', np.round(r0,2), ' via L1', np.round(r1,2), ' via L2', np.round(r2,2))
# OV positivity on T for each head (does head write the same token direction it reads? only meaningful within residual)
print()
for l in range(4):
    print('OV T-positivity L',l, [round(posit(ov(l,j)),2) for j in range(6)])
```