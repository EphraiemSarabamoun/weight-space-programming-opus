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
for seed in [1,2,3]:
    ica=FastICA(n_components=27, random_state=seed, max_iter=3000, whiten='unit-variance')
    S=ica.fit_transform(A); sk=skew(S,axis=0); S=S*np.sign(sk); M=ica.mixing_*np.sign(sk)
    eps=np.linalg.inv(M).T; Tok=E@eps
    mcol=np.linalg.norm(M,axis=0)
    sp=np.argsort(-mcol)[:2]
    cnt=(np.abs(S)>3).sum(0)
    Rc=F@Tok; Rp=F@O5@Tok
    Zc=(Rc-Rc.mean(1,keepdims=True))/Rc.std(1,keepdims=True)
    Zp=(Rp-Rp.mean(1,keepdims=True))/Rp.std(1,keepdims=True)
    J=np.array([[np.max(np.minimum(Zp[:,a],Zc[:,b])) for b in range(27)] for a in range(27)])
    flat=np.sort(J.ravel())
    pc=lambda v: round(100*np.searchsorted(flat,v)/flat.size,1)
    a,b=sp
    print(f"seed {seed}: top-|M| tokens {sp} |M| {np.round(mcol[sp],3)} next {np.round(np.sort(mcol)[-3],3)}; neuron counts {cnt[a]},{cnt[b]} (max other {np.max(np.delete(cnt,[a,b]))})")
    print(f"   J[{a},{a}] {J[a,a]:.2f} ({pc(J[a,a])}%)  J[{b},{b}] {J[b,b]:.2f} ({pc(J[b,b])}%)  J[{a}->{b}] {J[a,b]:.2f} ({pc(J[a,b])}%)  J[{b}->{a}] {J[b,a]:.2f} ({pc(J[b,a])}%)")
    # top 8 pairs overall
    o=np.argsort(-J.ravel())[:8]
    print("   top pairs (prev,cur):", [(int(i//27),int(i%27),round(float(J.ravel()[i]),1)) for i in o])
```