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
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
print("L0H4 OV restricted (out basis x in basis):\n", np.round(Sout.T@(Wo4@Wv4)@Sin,2))
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    print(f"L0H{j} QK on Sin:\n", np.round(Sin.T@Wq.T@Wk@Sin,3))
# top eigenplane of L0H1 and L0H5
for j in [1,5]:
    Wq,Wk,Wv,Wo=H[0][j]; M=Wq.T@Wk
    ev,evec=np.linalg.eig(M)
    o=np.argsort(-np.abs(ev))[:6]
    for t in o[:6:2]:
        v=evec[:,t]; P=np.stack([v.real,v.imag],1); Q,_=np.linalg.qr(P)
        print(j, np.round(ev[t],2), "overlap of plane with Sin", np.round(np.linalg.norm(Sin.T@Q,axis=0)**2,2))
```