```python
import numpy as np
from sklearn.decomposition import FastICA
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
g0=W['ln1_w.0']; g2=W['ln2_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
Traw=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
E=np.linalg.qr(Traw/g0[:,None])[0]
F=W['W_fc.0']*g2
A=F@E
ica=FastICA(n_components=27, random_state=0, max_iter=2000, whiten='unit-variance')
S=ica.fit_transform(A)   # 1536 x 27 sources (neuron loadings)
from scipy.stats import kurtosis
k=kurtosis(S,axis=0)
print("kurtosis of components", np.round(np.sort(k)[::-1],1))
# also kurtosis of PCA components for comparison
U,s,Vt=np.linalg.svd(A-A.mean(0),full_matrices=False)
print("PCA kurt", np.round(np.sort(kurtosis(U*s,axis=0))[::-1],1))
# skewness
from scipy.stats import skew
print("skew", np.round(skew(S,axis=0),1))
```