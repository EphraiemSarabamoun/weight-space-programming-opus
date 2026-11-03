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
X=F0@E
ica=FastICA(n_components=27,random_state=0,max_iter=3000,whiten='unit-variance')
S=ica.fit_transform(X); A=ica.mixing_
D=np.linalg.inv(A).T
ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
cur=ex                       # x-space current token dirs
prv=OV(0,5)@ex               # written by prev-token head
pr2=OV(0,1)@ex               # written by 2-back head
for name,Fm in [('MLP0',F0),('MLP1',W['W_fc.1']*W['ln2_w.1'])]:
    a=Fm@cur; b=Fm@prv; c=Fm@pr2
    # standardize per matrix
    a/=a.std(); b/=b.std(); c/=c.std()
    res=[]
    for t in range(27):
        m2=np.minimum(a[:,t],b[:,t]); m3=np.minimum(m2,c[:,t])
        res.append((t, np.sort(m2)[-3:].round(1), np.sort(m3)[-3:].round(1)))
    res.sort(key=lambda r:-r[2][-1])
    print(name, "token: top3 min(cur,prev) | top3 min(cur,prev,prev2)")
    for r in res[:10]: print("  ",r[0],r[1],r[2])
    print("   ...median token top m3:", np.median([r[2][-1] for r in res]).round(2))
    # baseline: mismatched tokens (cur t, prev s, prev2 u random)
    rng=np.random.default_rng(0); bl=[]
    for _ in range(200):
        t,s,u=rng.choice(27,3,replace=False)
        bl.append(np.max(np.minimum(np.minimum(a[:,t],b[:,s]),c[:,u])))
    print("   baseline random triples max m3: mean %.2f, 95pct %.2f, max %.2f"%(np.mean(bl),np.percentile(bl,95),np.max(bl)))
```