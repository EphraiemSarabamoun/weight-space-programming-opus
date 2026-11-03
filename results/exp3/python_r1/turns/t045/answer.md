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
Tok=E@eps   # keep scale (dual)
def OV(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g)
Rc=F@Tok              # response to current token (unnormalized; = S + const)
Rp=F@OV(0,5)@Tok      # response to previous token via L0H5
R2=F@OV(0,1)@Tok      # via L0H1 (1-3 back)
np.set_printoptions(linewidth=250, suppress=True)
for c in [13,19,21]:
    zc=(Rc[:,c]-Rc.mean(1))/Rc.std(1)
    idx=np.argsort(-zc)[:12]
    print(f"cur={c}: top detector neurons (z of cur within neuron)", np.round(zc[idx],1))
    for n in idx[:8]:
        zp=(Rp[n]-Rp[n].mean())/Rp[n].std()
        o=np.argsort(-zp)
        print(f"   n{n}: prev top {o[:4]} z {np.round(zp[o[:4]],1)}  prev bottom {o[-3:]} z {np.round(zp[o[-3:]],1)}   z(prev=13) {zp[13]:.1f} z(prev=19) {zp[19]:.1f}")
```