```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
g0=g[0]
U,S,Vt=np.linalg.svd(np.hstack([OVo(0,h) for h in [0,1,4,5]])); B=U[:,:2]
U3,S3,V3=np.linalg.svd(OVo(0,3)); A03=U3[:,:2]
print("A03 vs B overlap", np.linalg.svd(B.T@A03,compute_uv=False).round(2))
for l in [1,2,3]:
    for h in range(6):
        QK=blk('W_q',l,h).T@blk('W_k',l,h)
        Mb=B.T@QK@B
        ev=np.linalg.eigvals(Mb)
        b=W[f'ln1_b.{l}']
        cb=b@QK@B   # constant-query -> key B
        print(f"L{l}H{h} B->B block {Mb.round(2).tolist()} eig {np.round(ev,2)}  constq->B {cb.round(2)}")
```