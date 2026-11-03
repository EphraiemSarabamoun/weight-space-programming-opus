```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(1)
Q0,_=np.linalg.qr(rng.standard_normal((384,384)))
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    row=[]
    labels=[]
    for lw in range(l+1):
        for j in range(6):
            row.append(comp(F,OV(lw,j))); labels.append(f"A{lw}{j}")
    for lw in range(l):
        P=W[f'W_proj.{lw}']
        row.append(comp(F,P)); labels.append(f"M{lw}")
    base=np.mean([comp(F,Q0@OV(0,j)) for j in range(6)])
    print(f"MLP{l} reads:", " ".join(f"{a}:{b:.2f}" for a,b in zip(labels,row)), "base %.2f"%base)
# heads reading MLP outputs
for kind in 'qkv':
    for l in range(1,4):
        for j in range(6):
            A=W[f'W_{kind}.{l}'][sl(j)]*W[f'ln1_w.{l}']
            print(kind,f"L{l}H{j}"," ".join(f"M{lw}:{comp(A,W[f'W_proj.{lw}']):.2f}" for lw in range(l)), "base %.2f"%comp(A,Q0@W['W_proj.0']))
```