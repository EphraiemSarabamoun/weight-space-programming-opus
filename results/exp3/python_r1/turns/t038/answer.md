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
eps=np.linalg.inv(M).T   # columns: token vectors in E coords (dual)
Tok=E@eps                 # 384 x 27 token directions (x0 coords), arbitrary scale
Tok=Tok/np.linalg.norm(Tok,axis=0)
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H0=heads(0)
for j in range(6):
    Wq,Wk,Wv,Wo=H0[j]
    kp=np.linalg.norm(Wq.T@Wk@Tok,axis=0)
    qp=np.linalg.norm(Tok.T@Wq.T@Wk,axis=1)
    vp=np.linalg.norm(Wo@Wv@Tok,axis=0)
    print(f"L0H{j} key-pref {np.round(kp,2)}")
    print(f"      q-pref {np.round(qp,2)}")
    print(f"      v-norm {np.round(vp,2)}")
print("comp |S|>3 counts", (np.abs(S)>3).sum(0))
```