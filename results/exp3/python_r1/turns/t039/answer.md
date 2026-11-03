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
def OV(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g)
Rc=F@Tok; Rp=F@OV(0,5)@Tok; R2=F@OV(0,1)@Tok
Rc=(Rc-Rc.mean(0))/Rc.std(0); Rp=(Rp-Rp.mean(0))/Rp.std(0); R2=(R2-R2.mean(0))/R2.std(0)
print("corr check Rc vs S per comp", np.round([np.corrcoef(Rc[:,c],S[:,c])[0,1] for c in range(27)],2))
# For each token X: neurons with strongest current-X loading; their mean prev profile
for X in [13,19]:
    idx=np.argsort(-Rc[:,X])[:15]
    print(f"cur={X} top neurons: Rc", np.round(Rc[idx,X],1))
    print("   their prev profile (mean z):", np.round(Rp[idx].mean(0),1))
    idx=np.argsort(-Rp[:,X])[:15]
    print(f"prev={X} top neurons: their cur profile (mean z):", np.round(Rc[idx].mean(0),1))
# self-bigram: max over neurons of min(Rc[n,c],Rp[n,c])
sb=np.array([np.max(np.minimum(Rc[:,c],Rp[:,c])) for c in range(27)])
print("self-bigram strength per token", np.round(sb,1))
```