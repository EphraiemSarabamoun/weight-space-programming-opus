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
def cs(A,B): return np.sum(A*B)/np.linalg.norm(A)/np.linalg.norm(B)
for n in [6,12,24]:
    S=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
    U,Sv,Vt=np.linalg.svd(S,full_matrices=False); P=Vt[:n].T
    M5=P.T@(H[0][5][0].T@H[0][5][1])@P; R5,_=polar(M5)
    ev=np.linalg.eigvals(R5); print(n,"R5 eig angles", np.round(np.sort(np.degrees(np.angle(ev))),0))
    Ms=[P.T@(H[0][j][0].T@H[0][j][1])@P for j in range(6)]
    for j in [0,1,2,3,5]:
        print(f"  M{j}:", " ".join(f"{cs(Ms[j],np.linalg.matrix_power(R5,k)):+.2f}" for k in range(0,16)))
```