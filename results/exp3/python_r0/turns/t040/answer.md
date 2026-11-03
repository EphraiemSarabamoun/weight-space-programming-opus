```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
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
ica=FastICA(n_components=27,random_state=0,max_iter=5000,tol=1e-6,whiten='unit-variance')
S=ica.fit_transform(F0@E); A=ica.mixing_
sk=skew(S,axis=0); sg=np.sign(sk)
S=S*sg; A=A*sg
D=np.linalg.inv(A).T
ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
Wk0=W['W_k.0'][sl(0)]
k0=np.array([np.linalg.norm(Wk0@(g0*ex[:,t])) for t in range(27)])
for name,Fm in [('MLP0',F0),('MLP1',W['W_fc.1']*W['ln2_w.1'])]:
    a=Fm@ex; b=Fm@(OV(0,5)@ex); c=Fm@(OV(0,1)@ex)
    a/=a.std(); b/=b.std(); c/=c.std()
    rep=np.array([np.max(np.minimum(a[:,t],b[:,t])) for t in range(27)])
    rep2=np.array([np.max(np.minimum(a[:,t],c[:,t])) for t in range(27)])
    tri=np.array([np.max(np.minimum(np.minimum(a[:,t],b[:,t]),c[:,t])) for t in range(27)])
    Pm=np.array([[np.max(np.minimum(a[:,t],b[:,s])) for s in range(27)] for t in range(27)])
    off=Pm[~np.eye(27,dtype=bool)]
    rng=np.random.default_rng(1); bl=[]
    for _ in range(3000):
        t,s,u=rng.choice(27,3,replace=True)
        if t==s==u: continue
        bl.append(np.max(np.minimum(np.minimum(a[:,t],b[:,s]),c[:,u])))
    bl=np.array(bl)
    print(name, "pair off-diag 99pct %.2f max %.2f | triple non-identical 99pct %.2f max %.2f"%(np.percentile(off,99),off.max(),np.percentile(bl,99),bl.max()))
    o=np.argsort(-k0)
    print("  tok(K0norm) rep rep2 tri:", "  ".join(f"{t}({k0[t]:.2f}):{rep[t]:.1f}/{rep2[t]:.1f}/{tri[t]:.1f}" for t in o))
print("skew signs:",sg.astype(int))
```