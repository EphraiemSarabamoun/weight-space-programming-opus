```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,24)
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
B={j:P.T@(W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)])@P for j in range(6)}
n1=np.linalg.norm(B[1]); n5=np.linalg.norm(B[5])
rng=np.random.default_rng(3)
allres=[]
for trial in range(3):
    c=rng.standard_normal(2)
    Y=c[0]*B[1]/n1+c[1]*B[5]/n5
    ev,Wv=np.linalg.eig(Y)
    rows=[]
    for k in range(len(ev)):
        if ev[k].imag< -1e-8: continue
        w=Wv[:,k]/np.linalg.norm(Wv[:,k])
        l1=np.vdot(w,B[1]@w); l5=np.vdot(w,B[5]@w)
        # invariance quality
        q1=abs(l1)/np.linalg.norm(B[1]@w); q5=abs(l5)/np.linalg.norm(B[5]@w)
        rows.append((abs(l1),np.degrees(np.angle(l1)),q1,abs(l5),np.degrees(np.angle(l5)),q5, abs(ev[k].imag)>1e-8))
    rows.sort(key=lambda r:-(r[0]/n1+r[3]/n5))
    print("trial",trial)
    for r in rows[:14]:
        print("  H1 |l|=%5.2f th=%5.0f q=%.2f   H5 |l|=%5.2f th=%5.0f q=%.2f  complex=%s"%r)
```