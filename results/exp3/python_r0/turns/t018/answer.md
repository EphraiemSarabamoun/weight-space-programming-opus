```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
u4,s4,vt4=np.linalg.svd(OV(0,4)); u0,s0,vt0=np.linalg.svd(OV(0,0))
Pl=u4[:,:2]
print("A00 top2 in A04 plane coords:\n", (Pl.T@u0[:,:2]).round(2), s0[:2])
for l in [1,2,3]:
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        Wq=W[f'W_q.{l}'][sl(j)]; Wk=W[f'W_k.{l}'][sl(j)]
        u=(Wk*g).T@(Wq@b)
        c=Pl.T@u
        # also ln bias itself in plane
        M=(Wq*g).T@(Wk*g)
        # query-side vectors for the two plane key directions
        print(f"L{l}H{j} u in plane: {c.round(1)} angle {np.degrees(np.arctan2(c[1],c[0])):.0f} |u|={np.linalg.norm(u):.1f}")
print("ln1_b.l in plane:", [(Pl.T@W[f'ln1_b.{l}']).round(2) for l in [1,2,3]])
print("ln1_b norms", [np.linalg.norm(W[f'ln1_b.{l}']).round(2) for l in [1,2,3]])
```