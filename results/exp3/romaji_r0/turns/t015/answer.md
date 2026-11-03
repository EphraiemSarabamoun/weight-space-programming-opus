```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
F=W['W_fc.0']*W['ln2_w.0']; Fn=np.linalg.norm(F)**2
for name,A in [('Dscaled-P', np.vstack([blk('W_v',0,0),blk('W_v',0,3)])),
               ('raw', np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]]))]:
    if name=='Dscaled-P': A=A-A@P@P.T
    u,s,vt=np.linalg.svd(A)
    marg=[np.linalg.norm(F@vt[k])**2/Fn*384 for k in range(45)]
    print(name,'marginal MLP0 energy per dir (x random):', np.round(marg,1))
# also: marginal energy of H3 QK^T side etc of L3H3 induction in those directions
```