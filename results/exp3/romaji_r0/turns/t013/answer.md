```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j): 
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
_,_,Tv=np.linalg.svd(blk('W_v',0,3)); T=Tv[:18].T
print('P-T overlap (mean cos^2)', (np.linalg.svd(P.T@T,compute_uv=False)**2).round(2))
for j in range(6):
    for nm in ['W_q','W_k','W_v']:
        M=blk(nm,0,j)
        tot=np.linalg.norm(M)**2
        fp=np.linalg.norm(M@P)**2/tot; ft=np.linalg.norm(M@T)**2/tot
        # top-3 singular directions breakdown
        _,s,vt=np.linalg.svd(M)
        tops=[(round(float(s[i]),2), round(float(np.linalg.norm(P.T@vt[i])**2),2), round(float(np.linalg.norm(T.T@vt[i])**2),2)) for i in range(3)]
        print(f'L0H{j} {nm} frac in P {fp:.2f} in T {ft:.2f} (rand P {P.shape[1]/384:.2f}, T {18/384:.2f}) top sv (s, inP, inT):', tops)
```