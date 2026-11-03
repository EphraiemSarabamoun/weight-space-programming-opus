```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
b={l:W[f'ln1_b.{l}'] for l in range(4)}
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
# B subspace: stacked outputs of L0 heads 0,1,4,5
Bm=np.hstack([OVo(0,h) for h in [0,1,4,5]])
U,S,Vt=np.linalg.svd(Bm); print("B outputs sv",S[:6].round(2)); B=U[:,:2]
g0=g[0]
A=np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]); P=np.linalg.svd(A)[2][:12].T
for l in [1,2,3]:
    for h in range(6):
        s=slice(64*h,64*h+64)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]
        qc=Wq@b[l]            # constant query
        kc=Wk@b[l]
        # score contribution from constant query to keys: kdir = Wk^T diag(g) qc  (in residual space, before LN normalization)
        kdir=(Wk*g[l]).T@qc
        qdir=(Wq*g[l]).T@kc
        M=(Wq*g[l]).T@(Wk*g[l])
        Uq,Sq,Vk=np.linalg.svd(M)
        def fr(v,Sb): v=v/np.linalg.norm(v); return np.linalg.norm(Sb.T@v)**2
        print(f"L{l}H{h} |qconst|={np.linalg.norm(qc):.2f} |q-const->key dir|={np.linalg.norm(kdir):.2f} frac in B={fr(kdir,B):.2f} inP={fr(kdir,P):.2f}; topQK sv {Sq[:2].round(2)} q-side inB {fr(Uq[:,0],B):.2f} inP {fr(Uq[:,0],P):.2f}  k-side inB {fr(Vk[0],B):.2f} inP {fr(Vk[0],P):.2f} | cos(q-side, lnb)={abs(Uq[:,0]@b[l])/np.linalg.norm(b[l]):.2f}")
```