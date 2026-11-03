```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
def pos(A):
    ev=np.linalg.eigvals(A)
    return ev.real.sum()/np.abs(ev).sum(), np.abs(ev).max()
print("rows: reader head; cols: key-writer L0 heads 1,2,3,5 and none (M alone):  ratio(sum re/sum|ev|) and max|ev|")
for l in range(1,4):
    for j in range(6):
        M=Mqk(l,j)
        out=[]
        for a in [1,2,3,5]:
            r,m=pos(M@OV(0,a)); out.append(f"{r:+.2f}({m:.2f})")
        r,m=pos(M); out.append(f"self {r:+.2f}({m:.2f})")
        print(f"L{l}H{j}", "  ".join(out))
```