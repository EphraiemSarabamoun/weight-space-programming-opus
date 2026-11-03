```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
g0=W['ln1_w.0']; g2=W['ln2_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
Traw=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
E=np.linalg.qr(Traw/g0[:,None])[0]
F=W['W_fc.0']*g2
A=F@E
ica=FastICA(n_components=27, random_state=0, max_iter=3000, whiten='unit-variance')
S=ica.fit_transform(A)
sk=skew(S,axis=0); S=S*np.sign(sk)   # make heavy tail positive
M=ica.mixing_*np.sign(sk)            # A ~ S M^T + mean
# check one-hotness: for neurons with max |S| > 5, fraction of energy in top component
mx=np.abs(S).max(1); top=np.argmax(np.abs(S),1)
sel=mx>4
print("neurons with strong loading", sel.sum())
frac=(S[sel]**2).max(1)/(S[sel]**2).sum(1)
print("median energy fraction in top comp", np.median(frac))
cnt=np.bincount(top[sel],minlength=27)
print("neurons per component", cnt)
# number of strongly-loaded neurons per component (|S|>3)
print("count |S|>3 per comp", (np.abs(S)>3).sum(0))
np.save('/tmp/ica_M.npy',M); np.save('/tmp/ica_S.npy',S)
```