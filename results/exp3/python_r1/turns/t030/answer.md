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
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)   # 27-dim token subspace (raw coords)
# note: heads' matrices include g; T is in raw W_v row space = diag(g)*xhat-ish. Convert: token dirs in xhat space ~ diag(g) T
g0=W['ln1_w.0']
Tx=np.linalg.qr(g0[:,None]*T)[0]
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    M=Wq.T@Wk
    # key side restricted to token space, query full
    Mk=M@Tx   # 384x27
    U,S,Vt=np.linalg.svd(Mk,full_matrices=False)
    Mq=Tx.T@M
    U2,S2,Vt2=np.linalg.svd(Mq,full_matrices=False)
    print(f"L0H{j} key-on-T sv {np.round(S[:5],2)}  query-on-T sv {np.round(S2[:5],2)} | frac key in T {np.linalg.norm(Wk@Tx)**2/np.linalg.norm(Wk)**2:.2f} q in T {np.linalg.norm(Wq@Tx)**2/np.linalg.norm(Wq)**2:.2f} v in T {np.linalg.norm(Wv@Tx)**2/np.linalg.norm(Wv)**2:.2f}")
```