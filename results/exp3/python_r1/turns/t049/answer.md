```python
import numpy as np
from scipy.linalg import polar
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
for n in [16,24,32]:
    S=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
    U,Sv,Vt=np.linalg.svd(S,full_matrices=False); P=Vt[:n].T
    M5=P.T@(H[0][5][0].T@H[0][5][1])@P; R5,_=polar(M5)
    K=12
    Bs=np.stack([np.linalg.matrix_power(R5,k).ravel() for k in range(K)],1)
    print("n",n)
    for j in range(6):
        Mj=(P.T@(H[0][j][0].T@H[0][j][1])@P).ravel()
        coef,res,_,_=np.linalg.lstsq(Bs,Mj,rcond=None)
        fit=1-np.sum((Bs@coef-Mj)**2)/np.sum(Mj**2)
        print(f"  L0H{j} R2 {fit:.2f} coef (k=0..{K-1}):", np.round(coef/np.abs(coef).max(),2))
```