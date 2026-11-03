```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
OV02=OVo(0,2)
X=blk('W_q',3,1).T@blk('W_k',3,1)@OV02
ev=np.linalg.eigvals(X); ev=ev[np.argsort(-np.abs(ev))]
np.set_printoptions(linewidth=180,suppress=True)
print("eig X (L3H1 . L0H2):")
print(np.round(ev[:50],2))
print("sv X:", np.linalg.svd(X,compute_uv=False)[:50].round(2))
# symmetric part eigenvalues
Sx=(X+X.T)/2
es=np.linalg.eigvalsh(Sx)
print("sym part eig top:", np.sort(es)[::-1][:40].round(2))
print("sym part eig bottom:", np.sort(es)[:10].round(2))
```