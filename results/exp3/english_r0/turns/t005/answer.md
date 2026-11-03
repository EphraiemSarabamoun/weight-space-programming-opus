```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def Wq(l,h): s=slice(64*h,64*h+64); return W[f'W_q.{l}'][s]*g[l]
def Wk(l,h): s=slice(64*h,64*h+64); return W[f'W_k.{l}'][s]*g[l]
def Wv(l,h): s=slice(64*h,64*h+64); return W[f'W_v.{l}'][s]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s]
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
# baseline: random gaussian of same shapes
base=np.mean([comp(rng.normal(size=(64,384)),rng.normal(size=(384,384))) for _ in range(5)])
print("random baseline ~",base)
for l2 in range(1,4):
    for h2 in range(6):
        out=[]
        for l1 in range(l2):
            for h1 in range(6):
                O=OVo(l1,h1)
                out.append((l1,h1,comp(Wk(l2,h2),O),comp(Wq(l2,h2),O),comp(Wv(l2,h2),O)))
        ks=" ".join("%d.%d:%.2f"%(a,b,c) for a,b,c,d,e in out)
        qs=" ".join("%d.%d:%.2f"%(a,b,d) for a,b,c,d,e in out)
        print(f"L{l2}H{h2} K:",ks)
        print(f"L{l2}H{h2} Q:",qs)
```