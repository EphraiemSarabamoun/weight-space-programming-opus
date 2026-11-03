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
OV3=ov(0,3); OV0=ov(0,0)
print('cols: direct | Qcur-Kprev(H3) [induction] | Qprev(H3)-Kcur | Qprev-Kprev(H3) | Qcur-Kprev(H0) | Qprev(H0)-Kcur')
for l in [1,2,3]:
    for j in range(6):
        Q=qk(l,j)
        vals=[posit(Q), posit(Q@OV3), posit(OV3.T@Q), posit(OV3.T@Q@OV3), posit(Q@OV0), posit(OV0.T@Q)]
        print(f'L{l}H{j}', np.round(vals,2))
rng=np.random.default_rng(0)
b=[]
for t in range(50):
    Tr,_=np.linalg.qr(rng.standard_normal((384,24)))
    NT=Tr.T@(qk(1,0)@OV3)@Tr; b.append(np.trace(NT)/(np.sqrt(24)*np.linalg.norm(NT)))
print('baseline sd %.3f'%np.std(b))
```