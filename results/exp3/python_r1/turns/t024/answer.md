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
Ms=[H[0][j][0].T@H[0][j][1] for j in range(6)]
from scipy.linalg import polar
for n in [10,16,24,32]:
    S=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
    U,Sv,Vt=np.linalg.svd(S,full_matrices=False); P=Vt[:n].T
    M5=P.T@Ms[5]@P; R5,_=polar(M5)
    M1=P.T@Ms[1]@P; R1,_=polar(M1)
    def cs(A,B): return np.sum(A*B)/np.linalg.norm(A)/np.linalg.norm(B)
    out=[]
    for name,M in [('M1',M1),('M2',P.T@Ms[2]@P),('M3',P.T@Ms[3]@P),('M0',P.T@Ms[0]@P),('M5',M5)]:
        out.append(name+": "+" ".join(f"{cs(M,np.linalg.matrix_power(R5,k)):+.2f}" for k in range(0,6)))
    print(n, " | ".join(out))
    # also R1 vs powers of R5
    print("   R1 vs R5^k", " ".join(f"{cs(R1,np.linalg.matrix_power(R5,k)):+.2f}" for k in range(0,6)), " R5 vs R1^k", " ".join(f"{cs(R5,np.linalg.matrix_power(R1,k)):+.2f}" for k in range(0,4)))
```