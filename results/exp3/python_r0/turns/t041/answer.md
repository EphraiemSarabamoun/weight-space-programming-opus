```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
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
Wk0=W['W_k.0'][sl(0)]
for seed in [0,3,7]:
    ica=FastICA(n_components=27,random_state=seed,max_iter=5000,tol=1e-6,whiten='unit-variance')
    S=ica.fit_transform(F0@E); A=ica.mixing_
    sg=np.sign(skew(S,axis=0)); S*=sg; A=A*sg
    D=np.linalg.inv(A).T
    ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
    k0=np.array([np.linalg.norm(Wk0@(g0*ex[:,t])) for t in range(27)])
    o=np.argsort(-k0); T1,T2=o[0],o[1]   # 'token11' and 'token4' analogs
    a=F0@ex; b=F0@(OV(0,5)@ex); c=F0@(OV(0,1)@ex)
    a/=a.std(); b/=b.std(); c/=c.std()
    def P(t,s): return np.max(np.minimum(a[:,t],b[:,s]))
    def Tr(t,s,u): return np.max(np.minimum(np.minimum(a[:,t],b[:,s]),c[:,u]))
    allp=np.array([[P(t,s) for s in range(27)] for t in range(27)])
    print(f"seed {seed}: T1(top K0)={T1} T2={T2}; pair 99pct offdiag {np.percentile(allp[~np.eye(27,dtype=bool)],99):.1f}")
    print(f"   (cur,prev): T1,T1 {P(T1,T1):.1f} | T2,T2 {P(T2,T2):.1f} | cur T1 prev T2 {P(T1,T2):.1f} | cur T2 prev T1 {P(T2,T1):.1f}")
    print(f"   triples: T1T1T1 {Tr(T1,T1,T1):.1f} T2T2T2 {Tr(T2,T2,T2):.1f} T1(cur)T2T2 {Tr(T1,T2,T2):.1f} T2T1T1 {Tr(T2,T1,T1):.1f} T1T1T2 {Tr(T1,T1,T2):.1f} T2T2T1 {Tr(T2,T2,T1):.1f}")
    # rank of T1/T2 rows: best partners
    print("   cur=T1 best prev:", sorted([(round(float(allp[T1,s]),1),int(s)) for s in range(27)],reverse=True)[:3], " cur=T2 best prev:", sorted([(round(float(allp[T2,s]),1),int(s)) for s in range(27)],reverse=True)[:3])
    print("   prev=T1 best cur:", sorted([(round(float(allp[t,T1]),1),int(t)) for t in range(27)],reverse=True)[:3], " prev=T2 best cur:", sorted([(round(float(allp[t,T2]),1),int(t)) for t in range(27)],reverse=True)[:3])
```