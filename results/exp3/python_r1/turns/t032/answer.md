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
Traw=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
variants={'raw':Traw, 'div':np.linalg.qr(Traw/g0[:,None])[0], 'mul':np.linalg.qr(Traw*g0[:,None])[0]}
# alternative: token subspace directly from heads-matrices (W*g) rows: these act on xhat; their rowspace is diag(g0)^2 span(xhat)...
OV={j:H[0][j][3]@H[0][j][2] for j in range(6)}
def idcos(B): return np.trace(B)/np.linalg.norm(B)/np.sqrt(B.shape[0])
for nm,T in variants.items():
    print(nm)
    for l in range(1,4):
        print(f" L{l}:", " | ".join(" ".join(f"{idcos(T.T@(H[l][j][0].T@H[l][j][1])@OV[h]@T):+.2f}" for h in [5,1]) for j in range(6)),
              " || same:", " ".join(f"{idcos(T.T@(H[l][j][0].T@H[l][j][1])@T):+.2f}" for j in range(6)))
```