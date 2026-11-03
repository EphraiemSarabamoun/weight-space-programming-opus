```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
u4,s4,vt4=np.linalg.svd(OV(0,4)); S=u4[:,:2]
u0,s0,vt0=np.linalg.svd(OV(0,0))
print("A00 out dirs in S:",(S.T@u0[:,:2]).round(2),"(cols = A00 sv0, sv1)")
print("ln bias in S per layer:",[ (S.T@W[f'ln1_b.{l}']).round(2) for l in range(1,4)])
for l in [1,2]:
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        Wq=W[f'W_q.{l}'][sl(j)]; Wk=W[f'W_k.{l}'][sl(j)]
        M=(Wq*g).T@(Wk*g)
        BSS=S.T@M@S
        u=S.T@((Wk*g).T@(Wq@b))
        ev=np.linalg.eigvals(BSS)
        print(f"L{l}H{j} B_SS=\n{BSS.round(2)}  eig {ev.round(2)}  u_S {u.round(1)}")
```