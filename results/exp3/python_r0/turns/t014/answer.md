```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def topU(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k],s[:k]
U0,s0=topU(OV(0,0),3); U4,s4=topU(OV(0,4),3)
print("A00 sv",s0,"A04 sv",s4, "overlap A00/A04 planes", np.linalg.svd(U0[:,:2].T@U4[:,:2],compute_uv=False))
U,_=np.linalg.qr(np.concatenate([U0[:,:2],U4[:,:2]],1))
for l in [1,2,3]:
    g=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        Wq=W[f'W_q.{l}'][sl(j)]; Wk=W[f'W_k.{l}'][sl(j)]
        # bilinear in x-space (pre-LN normalized z): z^T (g Wq^T)(Wk g) z'
        M=(Wq*g).T@(Wk*g)
        qb=Wq@b; u=(Wk*g).T@qb
        blk=U.T@M@U
        # fraction of key-side energy: ||M U|| vs ||M||
        print(f"L{l}H{j} |u|={np.linalg.norm(u):.1f} frac u in U {np.linalg.norm(U.T@u)**2/np.linalg.norm(u)**2:.2f} | ||U^T M U||={np.linalg.norm(blk):.2f} ||M U||={np.linalg.norm(M@U):.2f} ||U^T M||={np.linalg.norm(U.T@M):.2f} ||M||={np.linalg.norm(M):.2f}")
        if l==1 and j<2: print(blk.round(3))
```