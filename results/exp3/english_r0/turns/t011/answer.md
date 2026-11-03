```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
g2={l:W[f'ln2_w.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
def outsub(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
srcs={}
for l in range(3):
    for h in range(6):
        srcs[f'A{l}.{h}']=OVo(l,h)
    srcs[f'M{l}']=W[f'W_proj.{l}']
readers={}
for l in range(1,4):
    for nm in ['W_q','W_k','W_v']:
        for h in range(6):
            readers[f'{nm[2]}{l}.{h}']=W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
    readers[f'fc{l}']=W[f'W_fc.{l}']*g2[l]
k=8
print("reader".ljust(7)+" ".join(s.rjust(5) for s in srcs))
for rn,A in readers.items():
    rl=int(rn[1]) if rn[0]!='f' else int(rn[2])
    row=[]
    for sn,S in srcs.items():
        sl=int(sn[1])
        if sl>=rl: row.append("    ."); continue
        U=outsub(S,k)
        row.append("%5.1f"%((np.linalg.norm(A@U)**2/np.linalg.norm(A)**2)/(k/384)))
    print(rn.ljust(7)+" ".join(row))
```