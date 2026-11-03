```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import skew
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
S=ica.fit_transform(A); sk=skew(S,axis=0); S=S*np.sign(sk); M=ica.mixing_*np.sign(sk)
eps=np.linalg.inv(M).T
Tok=E@eps; Tok=Tok/np.linalg.norm(Tok,axis=0)
s=slice(256,320); OV4=W['W_o.0'][:,s]@(W['W_v.0'][s]*g0)
U,Sv,Vt=np.linalg.svd(OV4)
print("OV4 sv", np.round(Sv[:6],2))
for k in range(5):
    proj=Vt[k]@Tok
    print(k, "input dir · Tok_c:", np.round(proj,2), " |in E|^2", round(float(np.linalg.norm(E.T@Vt[k])**2),2))
# Gram of Tok (cosines)
G=Tok.T@Tok
print("max off-diag cos", np.round(np.max(np.abs(G-np.eye(27))),2))
```