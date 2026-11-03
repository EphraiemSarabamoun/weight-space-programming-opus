```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,24)
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
B={j:P.T@(W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)])@P for j in range(6)}
S1=(B[1]+B[1].T)/2
es,V=np.linalg.eigh(S1)
o=np.argsort(es); es=es[o]; V=V[:,o]
for k in range(4):
    v=V[:,k]
    print(f"S1 eig {es[k]:.2f}: v^T B_h v for h=0..5:", [round(float(v@B[h]@v),2) for h in range(6)], " |B1 v - lam v|/|B1 v|=%.2f"%(np.linalg.norm(B[1]@v-es[k]*v)/np.linalg.norm(B[1]@v)))
```