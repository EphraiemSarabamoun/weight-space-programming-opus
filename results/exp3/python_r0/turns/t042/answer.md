```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']; b0=W['ln1_b.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
F0=W['W_fc.0']*W['ln2_w.0']
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
ica=FastICA(n_components=27,random_state=0,max_iter=5000,tol=1e-6,whiten='unit-variance')
S=ica.fit_transform(F0@E); A=ica.mixing_
sg=np.sign(skew(S,axis=0)); S*=sg; A=A*sg
D=np.linalg.inv(A).T
ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
a=F0@ex; b=F0@(OV(0,5)@ex); c=F0@(OV(0,1)@ex)
sa,sb,sc=a.std(),b.std(),c.std()
bias=W['W_fc.0']@W['ln2_b.0']
print("bias stats: mean %.3f std %.3f"%(bias.mean(),bias.std()))
for T in [11,4,12,8,10,13]:
    m=np.minimum(a[:,T]/sa,b[:,T]/sb); n=np.argmax(m)
    # compare: top single detectors for cur=T
    topcur=np.argsort(-a[:,T])[:5]; topprev=np.argsort(-b[:,T])[:5]
    print(f"tok {T}: repeat neuron {n}: cur {a[n,T]/sa:.1f} prev {b[n,T]/sb:.1f} prev2 {c[n,T]/sc:.1f} bias {bias[n]:.3f} | norm |w| {np.linalg.norm(F0[n]):.2f}")
    print("     top cur-detectors (cur,prev,bias):", [(round(float(a[k,T]/sa),1),round(float(b[k,T]/sb),1),round(float(bias[k]),2)) for k in topcur])
    print("     top prev-detectors (cur,prev,bias):", [(round(float(a[k,T]/sa),1),round(float(b[k,T]/sb),1),round(float(bias[k]),2)) for k in topprev])
```