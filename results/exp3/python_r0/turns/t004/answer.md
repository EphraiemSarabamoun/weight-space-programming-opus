```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
rng=np.random.default_rng(0)
def OV(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*W[f'ln1_w.{l}'])
def R(l,j,kind):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_{kind}.{l}'][s]*W[f'ln1_w.{l}']
def comp(A,B):  # reader A (r x 384), writer B (384 x c)
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
# baseline
Q0,_=np.linalg.qr(rng.standard_normal((384,384)))
writers=[(l,j) for l in range(3) for j in range(H)]
for kind in ['q','k','v']:
    print("=== reader", kind)
    for lr in range(1,4):
        for jr in range(H):
            A=R(lr,jr,kind)
            row=[]
            for (lw,jw) in writers:
                if lw<lr: row.append(comp(A,OV(lw,jw)))
                else: row.append(np.nan)
            base=np.mean([comp(A,Q0@OV(0,jw)) for jw in range(H)])
            print(f"L{lr}H{jr}", np.array(row), "base %.2f"%base)
```