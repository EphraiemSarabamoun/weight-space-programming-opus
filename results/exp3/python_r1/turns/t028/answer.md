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
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
OV={j:H[0][j][3]@H[0][j][2] for j in range(6)}
def idcos(B): return np.trace(B)/np.linalg.norm(B)/np.sqrt(B.shape[0])
for k in [16,24,28]:
    T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]]),k)
    print("k=",k)
    for l in range(1,4):
        row=[]
        for j in range(6):
            Wq,Wk,_,_=H[l][j]; M=Wq.T@Wk
            row.append(" ".join(f"{idcos(T.T@M@OV[h]@T):+.2f}" for h in [5,1,2,3,0]))
        print(f" L{l}:", " | ".join(row))
```