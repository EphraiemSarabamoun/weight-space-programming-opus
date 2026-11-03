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
g0=W['ln1_w.0']; top=np.argsort(-g0)[:40]
mask=np.zeros(384); mask[top]=1
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
us=[]
for j in range(6):
    Wq,Wk,_,_=H[1][j]
    M=Wq.T@(Wk@Sout)
    Uq,Sq,Vq=np.linalg.svd(M,full_matrices=False)
    u=Uq[:,0]; us.append(u)
    o=np.argsort(-np.abs(u))[:8]
    print(f"L1H{j} q-dir top coords {o} {np.round(u[o],2)} frac in top40 pos dims {(u[top]**2).sum():.2f}  PR {1/np.sum(u**4):.1f}  Vq {np.round(Vq[0],2)}")
us=np.array(us); print(np.round(np.abs(us@us.T),2))
print("Sout coords: ", np.round(Sout[[273,21,123,307,51]],2))
```