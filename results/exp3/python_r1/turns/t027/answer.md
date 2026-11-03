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
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]]),24)
OV5=H[0][5][3]@H[0][5][2]; OV1=H[0][1][3]@H[0][1][2]
def idcos(B): return np.trace(B)/np.linalg.norm(B)/np.sqrt(B.shape[0])
rng=np.random.default_rng(1)
for l in range(1,4):
    for j in range(6):
        Wq,Wk,_,_=H[l][j]; M=Wq.T@Wk
        same=T.T@M@T                 # q current token vs k current token
        ind5=T.T@M@OV5@T             # q current token vs k prev token (via L0H5)
        ind1=T.T@M@OV1@T             # vs token 2 back
        rev=T.T@OV5.T@M@T            # q prev token vs k current token
        print(f"L{l}H{j} same {idcos(same):+.2f} (|{np.linalg.norm(same):.1f}|)  ind5 {idcos(ind5):+.2f} (|{np.linalg.norm(ind5):.1f}|)  ind1 {idcos(ind1):+.2f} (|{np.linalg.norm(ind1):.1f}|)  qprev-kcur {idcos(rev):+.2f} (|{np.linalg.norm(rev):.1f}|)")
```