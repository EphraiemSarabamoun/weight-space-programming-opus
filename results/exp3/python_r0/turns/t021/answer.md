```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,24)
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
# P orthogonal to T
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
for j in [1,5,2,3,0]:
    M=W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)]
    B=P.T@M@P
    S=(B+B.T)/2; A=(B-B.T)/2
    es=np.linalg.eigvalsh(S); es=es[np.argsort(-np.abs(es))]
    ea=np.linalg.eigvals(A); ea=np.sort(np.abs(ea.imag))[::-1]
    print(f"L0H{j} |B|={np.linalg.norm(B):.2f} |M|={np.linalg.norm(M):.2f} |TMT|={np.linalg.norm(Th.T@M@Th):.2f}")
    print("   sym eig:", es[:16].round(2))
    print("   antisym |im|:", ea[:16:2].round(2))
```