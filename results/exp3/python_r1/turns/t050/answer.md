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
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,24)
OV={j:H[0][j][3]@H[0][j][2] for j in range(6)}
U4,S4,Vt4=np.linalg.svd(OV[4]); Sout=U4[:,:2]
def idcos(B): return np.trace(B)/np.linalg.norm(B)/np.sqrt(B.shape[0])
print("head  Sout_k  same  ind(prev)  ind(2back)  pp(qprev-kprev)  qprev-kcur  |QK|F  OVnorm")
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]; M=Wq.T@Wk
        fk=np.linalg.norm(Wk@Sout)**2/np.linalg.norm(Wk)**2*384/2
        same=idcos(T.T@M@T); i5=idcos(T.T@M@OV[5]@T); i1=idcos(T.T@M@OV[1]@T)
        pp=idcos(T.T@OV[5].T@M@OV[5]@T); qk=idcos(T.T@OV[5].T@M@T)
        print(f"L{l}H{j}  {fk:5.1f}  {same:+.2f}  {i5:+.2f}  {i1:+.2f}  {pp:+.2f}  {qk:+.2f}  {np.linalg.norm(M):.1f}  {np.linalg.norm(Wo@Wv):.1f}")
```