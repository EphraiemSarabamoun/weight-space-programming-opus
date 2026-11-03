```python
import numpy as np
from sklearn.decomposition import FastICA
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
F0=W['W_fc.0']*W['ln2_w.0']
for seed in [0,1,2]:
    ica=FastICA(n_components=27,random_state=seed,max_iter=3000,whiten='unit-variance')
    S=ica.fit_transform(F0@E); A=ica.mixing_
    D=np.linalg.inv(A).T
    ex=E@D; ex=ex/np.linalg.norm(ex,axis=0)
    def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
    a=F0@ex; b=F0@(OV(0,5)@ex); c=F0@(OV(0,1)@ex)
    a/=a.std(); b/=b.std(); c/=c.std()
    # pair matrix: P[t,s] = max_n min(a[n,t], b[n,s])  (cur t, prev s)
    Pm=np.array([[np.max(np.minimum(a[:,t],b[:,s])) for s in range(27)] for t in range(27)])
    diag=np.diag(Pm); off=Pm[~np.eye(27,dtype=bool)]
    print(f"seed {seed}: pair (cur,prev) off-diag: mean {off.mean():.2f} 99pct {np.percentile(off,99):.2f} max {off.max():.2f}; diag sorted:", np.sort(diag)[::-1][:6].round(1))
    # triple: same token at all three vs (t,t,s)/(t,s,t)
    tri=[np.max(np.minimum(np.minimum(a[:,t],b[:,t]),c[:,t])) for t in range(27)]
    order=np.argsort(tri)[::-1]
    print("   triple same sorted:", [(int(t),round(float(tri[t]),1)) for t in order[:5]])
    # random distinct triples baseline over many
    rng=np.random.default_rng(1); bl=[]
    for _ in range(2000):
        t,s,u=rng.choice(27,3,replace=True)
        if t==s==u: continue
        bl.append(np.max(np.minimum(np.minimum(a[:,t],b[:,s]),c[:,u])))
    bl=np.array(bl); print("   non-identical triples: 99pct %.2f max %.2f"%(np.percentile(bl,99),bl.max()))
    t4=order[0]
    # which tokens pair strongly with top token as prev
    print("   top token",t4,"row (cur=t4, prev=s) top:", sorted([(round(float(Pm[t4,s]),1),s) for s in range(27)],reverse=True)[:4], " col (cur=s, prev=t4):", sorted([(round(float(Pm[s,t4]),1),s) for s in range(27)],reverse=True)[:4])
```