```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
U,S,Vt=np.linalg.svd(W['W_proj.3'])
print("proj3 sv 18..34:",S[18:34].round(2))
for h in range(6):
    O=OVo(3,h)
    Uo,So,Vo=np.linalg.svd(O)
    print(f"L3H{h} OV sv:",So[:32].round(1))
# overlap of L3 head OV top outputs with proj3 top-k subspace
for k in [23,28,30]:
    Pk=U[:,:k]
    fr=[]
    for h in range(6):
        O=OVo(3,h); fr.append(np.linalg.norm(Pk.T@O)**2/np.linalg.norm(O)**2)
    print(f"frac of L3 head OV energy in proj3 top-{k} subspace:",np.round(fr,2), " random exp", round(k/384,3))
```