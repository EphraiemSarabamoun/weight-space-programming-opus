```python
import numpy as np
g0=W['ln1_w.0']
def blk(nm,h): return W[nm][64*h:64*h+64]*g0
# positional subspace guess from L0H2
A=np.vstack([blk('W_q.0',2),blk('W_k.0',2)])
U,S,Vt=np.linalg.svd(A)
print("L0H2 qk stacked sv",S[:20].round(2))
P=Vt[:14].T
def topread(M,k=1):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S[:k]
dirs={}
for h in [0,1,3,4,5]:
    for nm in ['W_q.0','W_k.0','W_v.0']:
        V,S=topread(blk(nm,h),2)
        dirs[(h,nm)]=V[:,0]
        print(h,nm,"top sv",S.round(2),"frac of top read dir in L0H2 pos subspace: %.2f"%(np.linalg.norm(P.T@V[:,0])**2), " 2nd: %.2f"%(np.linalg.norm(P.T@V[:,1])**2))
keys=list(dirs.keys())
D=np.array([dirs[k] for k in keys])
C=np.abs(D@D.T)
print(keys)
print(C.round(2))
```