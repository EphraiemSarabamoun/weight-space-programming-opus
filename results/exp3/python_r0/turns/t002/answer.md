```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H=6; hd=64
def heads(l):
    out=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]; Wv=W[f'W_v.{l}'][s]; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
for l in range(4):
    for j,(Wq,Wk,Wv,Wo) in enumerate(heads(l)):
        g=W[f'ln1_w.{l}']
        QK=(Wq*g).T@(Wk*g)
        OV=Wo@(Wv*g)
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        print(f"L{l}H{j} |q| {np.linalg.norm(Wq):.2f} |k| {np.linalg.norm(Wk):.2f} |v| {np.linalg.norm(Wv):.2f} |o| {np.linalg.norm(Wo):.2f} QK sv {sq[:6].round(2)} OV sv {so[:6].round(2)}")
```