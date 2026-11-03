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
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
g0=W['ln1_w.0']
Tx=np.linalg.qr(g0[:,None]*T)[0]
dirs={}
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]; M=Wq.T@Wk
    U,S,Vt=np.linalg.svd(M@Tx,full_matrices=False); dirs[f'k{j}']=Vt[0]*np.sign(Vt[0][np.argmax(np.abs(Vt[0]))])
    U2,S2,Vt2=np.linalg.svd(Tx.T@M,full_matrices=False); dirs[f'q{j}']=U2[:,0]*np.sign(U2[np.argmax(np.abs(U2[:,0])),0])
# the constant/bias query effect on token keys: bias query b
b0=W['ln1_b.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    qb=W['W_q.0'][s]@b0
    kt=qb@H[0][j][1]@Tx
    dirs[f'bk{j}']=kt/np.linalg.norm(kt); print(j,"bias-query key-token effect norm", np.linalg.norm(kt)/8)
names=list(dirs)
D=np.array([dirs[n] for n in names])
C=D@D.T
import pandas as pd
print(pd.DataFrame(np.round(C,2),index=names,columns=names).to_string())
```