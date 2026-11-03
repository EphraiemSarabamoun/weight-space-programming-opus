```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']; b0=W['ln1_b.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
F=W['W_fc.0']*W['ln2_w.0']
X=F@E
ica=FastICA(n_components=27,random_state=0,max_iter=3000,whiten='unit-variance')
S=ica.fit_transform(X)
A=ica.mixing_          # 27x27, X ~ S A^T + mean
k=kurtosis(S,axis=0)
# dual basis: token embedding coords e_t s.t. A_k . e_t = delta
D=np.linalg.inv(A).T   # columns d_t with A^T D = I  -> A[:,k].D[:,t]=delta
ex=E@D                  # x-space (unnormalized)
ex=ex/np.linalg.norm(ex,axis=0)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']; return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
rows=[]
Wq0=W['W_q.0'][sl(0)]; Wk0=W['W_k.0'][sl(0)]
qb0=Wq0@b0
for t in range(27):
    e=ex[:,t]; h=g0*e
    v5=np.linalg.norm(W['W_v.0'][sl(5)]@h); v1=np.linalg.norm(W['W_v.0'][sl(1)]@h); v4=np.linalg.norm(W['W_v.0'][sl(4)]@h)
    k0=qb0@(Wk0@h)
    nneu=np.sum(np.abs(S[:,t])>4); sgn=np.sign(S[np.argmax(np.abs(S[:,t])),t])
    anti=np.mean([e@Mqk(1,j)@e for j in range(1,6)])
    ind=np.mean([e@Mqk(l,j)@OV(0,5)@e for (l,j) in [(2,1),(3,1),(3,5)]])
    rows.append((t,k[t],nneu,v5,v1,v4,k0,anti,ind))
rows.sort(key=lambda r:-r[3])
print(" t  kurt nneu   V5    V1    V4   K0bias  L1self  induct")
for r in rows: print("%2d %6.0f %4d %5.2f %5.2f %5.2f %7.3f %7.3f %7.3f"%r)
```