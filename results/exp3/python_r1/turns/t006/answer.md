```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
print("random baseline", comp(rng.normal(size=(64,384)), rng.normal(size=(384,64))@rng.normal(size=(64,384))))
for l2 in range(1,4):
    for j2 in range(6):
        Wq,Wk,Wv,Wo=H[l2][j2]
        rowK=[];rowQ=[];rowV=[]
        for l1 in range(l2):
            for j1 in range(6):
                _,_,Wv1,Wo1=H[l1][j1]; OV=Wo1@Wv1
                rowK.append(comp(Wk,OV)); rowQ.append(comp(Wq,OV)); rowV.append(comp(Wv,OV))
        print(f"L{l2}H{j2} K:", " ".join(f"{x:.2f}" for x in rowK))
        print(f"      Q:", " ".join(f"{x:.2f}" for x in rowQ))
        print(f"      V:", " ".join(f"{x:.2f}" for x in rowV))
```