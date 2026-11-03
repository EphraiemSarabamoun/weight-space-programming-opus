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
def outsp(j,k,l=0):
    _,_,Wv,Wo=H[l][j]; U,S,Vt=np.linalg.svd(Wo@Wv); return U[:,:k]
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
T=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]]),24)  # token dims (raw residual)
O5=outsp(5,24); O1=outsp(1,24); O2=outsp(2,16); O3=outsp(3,16); So=outsp(4,2)
subs={'T':T,'O5':O5,'O1':O1,'O2':O2,'O3':O3,'Sout':So}
for a in subs:
    print(a, " ".join(f"{b}:{np.linalg.norm(subs[a].T@subs[b])**2:.1f}" for b in subs))
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,_=H[l][j]
        s=f"L{l}H{j} "
        for nm,B in subs.items():
            ch=B.shape[1]/384
            s+=f"{nm}: q{np.linalg.norm(Wq@B)**2/np.linalg.norm(Wq)**2/ch:.1f} k{np.linalg.norm(Wk@B)**2/np.linalg.norm(Wk)**2/ch:.1f} v{np.linalg.norm(Wv@B)**2/np.linalg.norm(Wv)**2/ch:.1f} | "
        print(s)
```