```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False)
    return vt[:k], s[:k]
names=[];subs=[]
for j in range(6):
    s_=slice(j*hd,(j+1)*hd)
    for kind in 'qkv':
        M=W[f'W_{kind}.0'][s_]
        vt,s=rs(M,12)
        print(f"H{j}{kind}", s.round(2))
        names.append(f"H{j}{kind}"); subs.append(vt[:8])
n=len(subs)
O=np.zeros((n,n))
for a in range(n):
    for b in range(n):
        O[a,b]=np.linalg.norm(subs[a]@subs[b].T)**2/8
print(names)
print(O.round(2))
```