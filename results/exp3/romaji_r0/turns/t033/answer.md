```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def ov(l,j):
    s_=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s_]@(W[f'W_v.{l}'][s_]*Dl(l))
U,s,_=np.linalg.svd(W['W_proj.3'])
U23=U[:,1:24]
O=ov(3,3)
uo,so,vo=np.linalg.svd(O)
print('L3H3 OV sv', so[:40].round(2))
print('frac of each OV output dir in Wproj3 token-subspace(1..23):', np.round([np.linalg.norm(U23.T@uo[:,k])**2 for k in range(36)],2))
# Also final-layer write subspace from W_o.3 all heads combined except massive: 
# energy of W_proj.2, W_proj.1 in U23 vs random
for nm in ['W_proj.0','W_proj.1','W_proj.2','W_proj.3','W_o.1','W_o.2','W_o.3']:
    M=W[nm]; print(nm, 'energy frac in U23 %.3f (rand %.3f)'%(np.linalg.norm(U23.T@M)**2/np.linalg.norm(M)**2, 23/384))
```