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
g0=W['ln1_w.0']
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
for v,name in [(Sin[:,0],'Sin0'),(Sin[:,1],'Sin1'),(Sout[:,0],'Sout0'),(Sout[:,1],'Sout1')]:
    o=np.argsort(-np.abs(v))[:8]
    print(name, "top coords", o, np.round(v[o],3), "ln1w0", np.round(g0[o],2), "PR", 1/np.sum(v**4))
_,_,Wv0,Wo0=H[0][0]
U0,S0,Vt0=np.linalg.svd(Wo0@Wv0)
print("L0H0 out vs L0H4 out", np.round(U0[:,:2].T@Sout,3))
print("L0H0 in vs Sin", np.round(Vt0[:2]@Sin,3))
```