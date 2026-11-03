```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
g2={l:W[f'ln2_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
def outsub(l,h,k):
    U,S,Vt=np.linalg.svd(OVo(l,h)); return U[:,:k]
readers={}
for l in range(4):
    for nm in ['W_q','W_k','W_v']:
        for h in range(6):
            readers[f'{nm}.{l}H{h}']=W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
    readers[f'W_fc.{l}']=W[f'W_fc.{l}']*g2[l]
for (l0,h0,k) in [(0,2,20),(0,0,2),(0,5,2),(0,3,3)]:
    U=outsub(l0,h0,k)
    res=[]
    for name,A in readers.items():
        lname=int(name.split('.')[1][0])
        if lname<l0 or (lname==l0 and not name.startswith('W_fc')): continue
        frac=(np.linalg.norm(A@U)**2/np.linalg.norm(A)**2)/(k/384)
        res.append((name,frac))
    print(f"L{l0}H{h0} out subspace k={k}:"," ".join("%s:%.1f"%(n,f) for n,f in res))
```