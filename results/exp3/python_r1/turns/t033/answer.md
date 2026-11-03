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
g0=W['ln1_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
for l in range(1,4):
    for j in range(6):
        B=T.T@(H[l][j][0].T@H[l][j][1])@T
        Bs=(B+B.T)/2
        ev=np.linalg.eigvalsh(Bs)
        sv=np.linalg.svd(B,compute_uv=False)
        print(f"L{l}H{j} sym-eigs low {np.round(ev[:4],2)} high {np.round(ev[-4:],2)}  sv {np.round(sv[:5],2)}")
```