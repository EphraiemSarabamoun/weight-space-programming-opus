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
np.set_printoptions(linewidth=250, suppress=True)
for j in [0,1,2,3,5]:
    s=slice(64*j,64*j+64)
    Wq=W['W_q.0'][s]*g0; Wk=W['W_k.0'][s]*g0
    TT=Tok.T@Wq.T@Wk@Tok
    sc=np.abs(TT).max()
    TT=TT/sc
    print(f"L0H{j} scale {sc:.4f}; diag (same-token):", np.round(np.diag(TT),2))
    print("   row13:", np.round(TT[13],2))
    print("   row19:", np.round(TT[19],2))
    print("   col13:", np.round(TT[:,13],2))
    print("   col19:", np.round(TT[:,19],2))
```