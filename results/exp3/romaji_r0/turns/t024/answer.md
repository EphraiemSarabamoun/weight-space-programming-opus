```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
F=W['W_fc.0']*W['ln2_w.0']
Pr=W['W_proj.0']
ns=np.argsort(-Pr[184])[:8]
Fn=F[ns]/np.linalg.norm(F[ns],axis=1,keepdims=True)
m=Fn.mean(0); m/=np.linalg.norm(m)
print('m energy in T %.3f P %.3f (rand %.3f %.3f)'%(np.linalg.norm(T.T@m)**2, np.linalg.norm(P.T@m)**2, 24/384, P.shape[1]/384))
# per-T-direction
print('m comps on T dirs', (T.T@m).round(2))
# key directions of L0H1,H2,H4,H5 top sv
for j in [1,2,4,5]:
    M=blk('W_k',0,j); _,sv,v=np.linalg.svd(M)
    for i in range(2):
        print(f'L0H{j} K dir{i} sv {sv[i]:.2f}: in T {np.linalg.norm(T.T@v[i])**2:.2f} in P {np.linalg.norm(P.T@v[i])**2:.2f} cos m {v[i]@m:.2f}')
```