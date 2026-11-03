```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; OV02=O@V
U,S,Vt=np.linalg.svd(Q.T@K)
P=orth(np.hstack([U[:,:26],Vt[:26].T]),30)
Uv,Sv,Vv=np.linalg.svd(OV02); T=orth(Vv[:24].T-P@(P.T@Vv[:24].T),20)
Tr=orth(Vv[:24].T,24)  # raw input subspace of 0.2 values
def evtr(M):
    ev=np.linalg.eigvals(M); return np.real(ev).sum()/np.abs(ev).sum(), np.abs(ev).max()
rng=np.random.default_rng(0)
for l in [1,2,3]:
    for h in range(6):
        Q,K,V,O=H[(l,h)]
        M=Q.T@K
        a=evtr(Tr.T@M@Tr); b=evtr(Tr.T@M@OV02@Tr)
        # baseline: random rotation of OV02 output
        R=np.linalg.qr(rng.normal(size=(384,384)))[0]
        c=evtr(Tr.T@M@R@OV02@Tr)
        print(l,h,'same-token tr %.2f max %.2f | induction(0.2) tr %.2f max %.2f | rand tr %.2f max %.2f'%(a+b+c))
```