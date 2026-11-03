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
def OV(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g)
O5=OV(0,5)
ica=FastICA(n_components=27, random_state=1, max_iter=3000, whiten='unit-variance')
S=ica.fit_transform(A); sk=skew(S,axis=0); S=S*np.sign(sk); M=ica.mixing_*np.sign(sk)
eps=np.linalg.inv(M).T; Tok=E@eps
Rc=F@Tok; Rp=F@O5@Tok
Zc=(Rc-Rc.mean(1,keepdims=True))/Rc.std(1,keepdims=True)
Zp=(Rp-Rp.mean(1,keepdims=True))/Rp.std(1,keepdims=True)
J=np.array([[np.max(np.minimum(Zp[:,a],Zc[:,b])) for b in range(27)] for a in range(27)])
d=np.diag(J); off=J[~np.eye(27,dtype=bool)]
print("diag J", np.round(d,1)); print("offdiag pct", np.round(np.percentile(off,[50,90,99]),2), "diag median", np.median(d))
cor=[np.corrcoef(Zc[:,c],Zp[:,c])[0,1] for c in range(27)]
print("corr(Zc,Zp) same token", np.round(cor,2))
# OV5 action on token space: Tok^T OV5 Tok (normalized)
Tn=Tok/np.linalg.norm(Tok,axis=0)
B=Tn.T@O5@Tn
print("OV5 in token coords: diag mean", B.diagonal().mean(), "offdiag abs mean", np.abs(B[~np.eye(27,dtype=bool)]).mean())
cnt=(np.abs(S)>3).sum(0); print("neuron counts", cnt)
```