```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def QKmat(l,j):
    s=slice(64*j,64*j+64); wl=Dl(l)
    return (W[f'W_q.{l}'][s]*wl).T@(W[f'W_k.{l}'][s]*wl)
QK3=QKmat(0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
Sd=[21,39,58,99,184,383]
E=np.eye(384)[:,Sd]
for l in range(4):
    for j in range(6):
        M=QKmat(l,j)
        pp=np.linalg.norm(P.T@M@P); tt=np.linalg.norm(T.T@M@T); tp=np.linalg.norm(T.T@M@P); pt=np.linalg.norm(P.T@M@T)
        sk=np.linalg.norm(M@E); tot=np.linalg.norm(M)
        print(f'L{l}H{j} tot {tot:6.2f} PP {pp:5.2f} TT {tt:5.2f} TqPk {tp:5.2f} PqTk {pt:5.2f} keySink {sk:5.2f}')
```