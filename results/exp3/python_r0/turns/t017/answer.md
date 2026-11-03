```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th=topV(Vst,27)
def pr(A):
    ev=np.linalg.eigvals(A); return ev.real.sum()/np.abs(ev).sum()
for eps in [0.0,0.05,0.2]:
    E,_=np.linalg.qr(Th/(g0[:,None]+eps))
    print("eps",eps)
    for a in [1,2,3,5]:
        Bo=E.T@OV(0,a)@E
        print(f"  OV0{a} on E: pos ratio {pr(Bo):+.2f}, |.| {np.linalg.norm(Bo):.2f} vs full {np.linalg.norm(OV(0,a)):.2f}; frac of OV output in E {np.linalg.norm(E.T@OV(0,a))**2/np.linalg.norm(OV(0,a))**2:.2f}")
    for l in range(1,4):
        for j in range(6):
            M=Mqk(l,j)
            s=[f"self {pr(E.T@M@E):+.2f}({np.linalg.norm(E.T@M@E):.2f})"]
            for a in [1,5,2,3]:
                B=E.T@M@OV(0,a)@E
                s.append(f"o{a} {pr(B):+.2f}({np.linalg.norm(B):.2f})")
            print(f"  L{l}H{j}", "  ".join(s))
```