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
S=ica.fit_transform(X); A=ica.mixing_
k=kurtosis(S,axis=0)
D=np.linalg.inv(A).T
ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
# Gram of token directions (are they near-orthogonal?)
G=ex.T@ex
print("max |off-diag| cos among token dirs:", np.abs(G-np.eye(27)).max().round(2), "mean", np.abs(G-np.eye(27)).mean().round(3))
for t in [11,4,19,16,13,1]:
    s=S[:,t]; top=np.argsort(-np.abs(s))[:12]
    print(f"tok{t}: kurt {k[t]:.0f}, top |s|:", s[top].round(1))
# key preferences in L0 heads per token: use query = bias + top-3 query singular dirs
for j in [0,2,3]:
    Wq=W['W_q.0'][sl(j)]; Wk=W['W_k.0'][sl(j)]
    M=Wq.T@Wk
    kt=np.array([Wk@(g0*ex[:,t]) for t in range(27)])   # key vecs per token (64)
    qb=Wq@b0
    u,s,vt=np.linalg.svd(Wq,full_matrices=False)
    # score spread: for random positional-ish queries use top 5 query output dirs
    sc=kt@u[:,:5]*s[:5]   # token x 5
    print(f"L0H{j} bias pref per token (sorted):", sorted([(round(float(kt[t]@qb),3),t) for t in range(27)],reverse=True)[:6])
    nrm=np.linalg.norm(kt,axis=1)
    print(f"   key norm per token top:", sorted([(round(float(nrm[t]),2),t) for t in range(27)],reverse=True)[:6])
```