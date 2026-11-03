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
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
for l in range(1,3):
    b=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq,Wk,Wv,Wo=H[l][j]
        qb=W[f'W_q.{l}'][s]@b   # constant query part
        kS=Wk@Sout               # 64x2
        bias_eff=qb@kS/8         # per unit of Sout coordinate (xhat units)
        M=Wq.T@kS  # 384x2
        Uq,Sq,Vq=np.linalg.svd(M,full_matrices=False)
        # overlap of top query dir with Sin, with Sout, with ln1_b direction
        bn=b/np.linalg.norm(b)
        print(f"L{l}H{j} bias-query . Wk Sout /8 = {np.round(bias_eff,2)}; |qb| {np.linalg.norm(qb):.2f}; topq dir overlap Sin {np.linalg.norm(Sin.T@Uq[:,0])**2:.2f} Sout {np.linalg.norm(Sout.T@Uq[:,0])**2:.2f}; sv {np.round(Sq,2)}")
```