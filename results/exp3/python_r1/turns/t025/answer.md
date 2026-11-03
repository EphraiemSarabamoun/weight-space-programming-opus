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
n=24
S=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,Sv,Vt=np.linalg.svd(S,full_matrices=False); P=Vt[:n].T
M5=P.T@(H[0][5][0].T@H[0][5][1])@P; R5,_=polar(M5)
def cs(A,B): return np.sum(A*B)/np.linalg.norm(A)/np.linalg.norm(B)
Rp=[np.linalg.matrix_power(R5,k) for k in range(0,8)]
for l in range(4):
    for j in range(6):
        Wq,Wk,_,_=H[l][j]; M=Wq.T@Wk
        MP=P.T@M@P
        frac=np.linalg.norm(MP)/np.linalg.norm(M)
        # fraction of query/key norm in P
        fq=np.linalg.norm(Wq@P)**2/np.linalg.norm(Wq)**2; fk=np.linalg.norm(Wk@P)**2/np.linalg.norm(Wk)**2
        print(f"L{l}H{j} |MP|/|M| {frac:.2f} q-in-P {fq:.2f} k-in-P {fk:.2f}  cos with R5^k:", " ".join(f"{cs(MP,R):+.2f}" for R in Rp))
```