```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H=6; hd=64
for l in range(4):
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g
        qb=W[f'W_q.{l}'][s]@b; kb=W[f'W_k.{l}'][s]@b
        u=Wk.T@qb   # key preference vector (in z space)
        A=Wk@Wq.T   # 64x64 same nonzero eigs as Wq^T Wk
        ev=np.linalg.eigvals(A)
        ev=ev[np.argsort(-np.abs(ev))]
        ang=np.angle(ev[:12])
        print(f"L{l}H{j} |qb| {np.linalg.norm(qb):.2f} |u| {np.linalg.norm(u):.2f} top eig abs {np.abs(ev[:10]).round(2)} angle(deg) {(ang*180/np.pi).round(0)}")
        print("    sum eig real %.2f  sum |eig| %.2f" % (ev.real.sum(), np.abs(ev).sum()))
```