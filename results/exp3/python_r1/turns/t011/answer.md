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
print("Sout vs Sin", np.round(Sout.T@Sin,3))
# Note: layer>0 heads have their own LN weights; Sin defined in raw residual basis (after ln1_w.0 scaling? no: Wv4 includes g0)
# Wv4 = W_v * g0, so Sin is direction in xhat space (normalized residual, before gain). Residual raw basis same as xhat basis up to scalar.
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        fq=np.linalg.norm(Wq@Sin)**2/np.linalg.norm(Wq)**2*384/2
        fk=np.linalg.norm(Wk@Sin)**2/np.linalg.norm(Wk)**2*384/2
        fv=np.linalg.norm(Wv@Sin)**2/np.linalg.norm(Wv)**2*384/2
        M=Sin.T@Wq.T@Wk@Sin
        print(f"L{l}H{j} enrich q {fq:.1f} k {fk:.1f} v {fv:.1f}  QK|Sin {np.round(M,2).tolist()}")
```