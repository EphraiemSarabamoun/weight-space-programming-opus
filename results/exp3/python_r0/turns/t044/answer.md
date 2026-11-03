```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
F0=W['W_fc.0']*W['ln2_w.0']
ica=FastICA(n_components=27,random_state=0,max_iter=5000,tol=1e-6,whiten='unit-variance')
S=ica.fit_transform(F0@E); A=ica.mixing_
sg=np.sign(skew(S,axis=0)); S*=sg; A=A*sg
D=np.linalg.inv(A).T
ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
a=F0@ex; b=F0@(OV(0,5)@ex)
u4,s4,vt4=np.linalg.svd(OV(0,4)); Sp=u4[:,:2]
bias=W['W_fc.0']@W['ln2_b.0']
sp=F0@Sp    # neuron x 2: response to unit S-plane dirs
print("S-plane read: std over neurons", sp.std(0).round(3), " |row| mean %.2f"%np.linalg.norm(F0,axis=1).mean())
# typical S content direction: mean output direction of L0H4 = it's u4[:,0] scaled. assume s ~ c*(cos,sin) unknown; print projections
for n,T in [(1465,11),(524,4),(1214,12),(360,8),(213,10)]:
    topcur=np.argsort(-a[:,T])[:4]
    print(f"tok {T} repeat n{n}: S-read {sp[n].round(2)} bias {bias[n]:.2f} | single cur-detectors S-read:", [sp[k].round(2).tolist() for k in topcur if k!=n][:3], "bias", [round(float(bias[k]),2) for k in topcur if k!=n][:3])
# Distribution of S-plane read along the L0H4 dominant output direction for all neurons
print("all neurons S-read along dir0: pct5 %.2f median %.2f pct95 %.2f"%tuple(np.percentile(sp[:,0],[5,50,95])))
print("all neurons S-read along dir1: pct5 %.2f median %.2f pct95 %.2f"%tuple(np.percentile(sp[:,1],[5,50,95])))
```