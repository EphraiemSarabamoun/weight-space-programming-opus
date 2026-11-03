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
F0=W['W_fc.0']*W['ln2_w.0']
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']; return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
Wq0=W['W_q.0'][sl(0)]; Wk0=W['W_k.0'][sl(0)]
def analyze(seed):
    ica=FastICA(n_components=27,random_state=seed,max_iter=5000,tol=1e-6,whiten='unit-variance')
    S=ica.fit_transform(F0@E); A=ica.mixing_
    D=np.linalg.inv(A).T
    ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
    a=F0@ex; b=F0@(OV(0,5)@ex); c=F0@(OV(0,1)@ex)
    a/=a.std(); b/=b.std(); c/=c.std()
    rep=np.array([np.max(np.minimum(a[:,t],b[:,t])) for t in range(27)])
    rep2=np.array([np.max(np.minimum(a[:,t],c[:,t])) for t in range(27)])
    k0=np.array([np.linalg.norm(Wk0@(g0*ex[:,t])) for t in range(27)])
    v4=np.array([np.linalg.norm(W['W_v.0'][sl(4)]@(g0*ex[:,t])) for t in range(27)])
    v5=np.array([np.linalg.norm(W['W_v.0'][sl(5)]@(g0*ex[:,t])) for t in range(27)])
    ind=np.array([np.mean([ex[:,t]@Mqk(l,j)@OV(0,5)@ex[:,t] for (l,j) in [(2,1),(3,1),(3,5)]]) for t in range(27)])
    k3b=np.array([ (W['W_q.0'][sl(3)]@b0)@(W['W_k.0'][sl(3)]@(g0*ex[:,t])) for t in range(27)])
    kur=kurtosis(S,axis=0)
    o=np.argsort(-k0)
    print(f"seed {seed}")
    print(" tok  K0norm  V4    V5   induct  rep(cur,prev) rep(cur,prev2)  K3bias  kurt")
    for t in o: print(f"  {t:2d}  {k0[t]:.3f}  {v4[t]:.3f} {v5[t]:.3f} {ind[t]:6.3f}   {rep[t]:5.1f}   {rep2[t]:5.1f}   {k3b[t]:6.3f}  {kur[t]:5.0f}")
analyze(0)
analyze(5)
```