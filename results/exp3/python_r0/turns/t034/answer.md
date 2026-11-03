```python
import numpy as np
from sklearn.decomposition import FastICA
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
F=W['W_fc.0']*W['ln2_w.0']
X=F@E    # 1536 x 27 coordinates
print("energy frac of fc0 rows in E: %.3f"%(np.linalg.norm(X)**2/np.linalg.norm(F)**2))
# also try h-space T with layer0 V/Q/K rows? use fc0 in x-space E.
from scipy.stats import kurtosis
for seed in [0,1]:
    ica=FastICA(n_components=27,random_state=seed,max_iter=2000,whiten='unit-variance')
    S=ica.fit_transform(X)   # sources per neuron
    k=kurtosis(S,axis=0)
    print("seed",seed,"kurtosis sorted:",np.sort(k)[::-1].round(1))
# compare: kurtosis of PCA components
u,s,vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
print("PCA comp kurtosis:",np.sort(kurtosis(u*s,axis=0))[::-1].round(1))
```