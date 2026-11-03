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
Tok=E@eps
def OV(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g)
Rc=F@Tok; Rp=F@OV(0,5)@Tok
Zc=(Rc-Rc.mean(1,keepdims=True))/Rc.std(1,keepdims=True)
Zp=(Rp-Rp.mean(1,keepdims=True))/Rp.std(1,keepdims=True)
J=np.zeros((27,27))  # J[prev,cur]
for a in range(27):
    for b in range(27):
        J[a,b]=np.max(np.minimum(Zp[:,a],Zc[:,b]))
np.set_printoptions(linewidth=250, suppress=True)
print("overall J distribution percentiles", np.round(np.percentile(J,[5,25,50,75,95]),2))
print("prev=13 row (cur=*):", np.round(J[13],1))
print("prev=19 row (cur=*):", np.round(J[19],1))
print("cur=13 col (prev=*):", np.round(J[:,13],1))
print("cur=19 col (prev=*):", np.round(J[:,19],1))
print("J[13,13]",J[13,13]," J[19,19]",J[19,19]," J[13,19] (prev13,cur19)",J[13,19]," J[19,13]",J[19,13])
# rank of these among all
flat=np.sort(J.ravel())
for nm,v in [('13,13',J[13,13]),('19,19',J[19,19]),('13->19',J[13,19]),('19->13',J[19,13])]:
    print(nm, "percentile", round(100*np.searchsorted(flat,v)/flat.size,1))
```