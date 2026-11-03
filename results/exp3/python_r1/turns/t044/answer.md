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
res=[]
for seed in [0,1,2]:
    ica=FastICA(n_components=27, random_state=seed, max_iter=3000, whiten='unit-variance')
    S=ica.fit_transform(A); sk=skew(S,axis=0); S=S*np.sign(sk); M=ica.mixing_*np.sign(sk)
    res.append(M)
# check stability: match components across seeds by abs cosine of mixing columns
def ncol(M): return M/np.linalg.norm(M,axis=0)
C=np.abs(ncol(res[0]).T@ncol(res[1])); print("seed0 vs seed1 best match cos", np.round(np.sort(C.max(1)),2))
M=res[0]
eps=np.linalg.inv(M).T
Tok=E@eps; Tok=Tok/np.linalg.norm(Tok,axis=0)
print("|M col| (neuron weight scale per token)", np.round(np.linalg.norm(M,axis=0),2))
print("|eps col|", np.round(np.linalg.norm(eps,axis=0),2))
# token specialness across all layer0 heads: norm of row and col in token-token QK, normalized by mean
np.set_printoptions(linewidth=250, suppress=True)
for j in [0,1,2,3,5]:
    s=slice(64*j,64*j+64)
    TT=Tok.T@(W['W_q.0'][s]*g0).T@(W['W_k.0'][s]*g0)@Tok
    TT=TT/np.abs(TT).mean()
    print(f"L0H{j} rownorm", np.round(np.linalg.norm(TT,axis=1)/np.sqrt(27),1))
    print(f"      colnorm", np.round(np.linalg.norm(TT,axis=0)/np.sqrt(27),1))
```