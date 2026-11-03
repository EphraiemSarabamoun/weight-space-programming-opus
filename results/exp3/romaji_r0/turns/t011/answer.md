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
_,sv,Vt=np.linalg.svd(W['W_v.0'][192:256]*Dl(0))
def posit(N,T):
    NT=T.T@N@T
    return np.trace(NT)/(np.sqrt(T.shape[1])*np.linalg.norm(NT))
rng=np.random.default_rng(0)
for r in [12,18,24]:
    T=Vt[:r].T
    print('r=',r)
    for l in [1,2,3]:
        direct=[];induct=[];
        for j in range(6):
            Q=qk(l,j)
            direct.append(posit(Q,T)); induct.append(posit(Q@OV3,T))
        print(f' L{l} direct', np.round(direct,2), ' induct(K via L0H3)', np.round(induct,2))
    # baseline random subspaces
    b=[]
    for t in range(30):
        Tr,_=np.linalg.qr(rng.standard_normal((384,r)))
        b.append(posit(qk(1,0)@OV3,Tr))
    print(' baseline random T: mean %.2f sd %.2f'%(np.mean(b),np.std(b)))
# also L0 heads direct matching in T
T=Vt[:18].T
print('L0 direct', [round(posit(qk(0,j),T),2) for j in range(6)])
```