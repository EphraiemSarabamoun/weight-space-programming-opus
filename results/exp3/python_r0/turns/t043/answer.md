```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import skew
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']
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
a=F0@ex; b=F0@(OV(0,5)@ex)
print("raw std a %.3f b %.3f"%(a.std(),b.std()))
bias=W['W_fc.0']@W['ln2_b.0']
for n,T in [(1465,11),(524,4),(1214,12),(360,8),(213,10)]:
    print(n,T,"raw cur %.3f prev %.3f bias %.3f  |w| %.2f"%(a[n,T],b[n,T],bias[n],np.linalg.norm(F0[n])))
# how big can a token's projection be? OV(0,5) maps unit e to vector of norm:
print("OV05 gain on unit token dirs: mean %.2f"%np.linalg.norm(OV(0,5)@ex,axis=0).mean())
# compare typical magnitude of residual contributions: the S-plane writer OV04 on unit positional dir is 13.9
```