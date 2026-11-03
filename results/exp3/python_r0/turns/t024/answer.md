```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,24)
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
b0=W['ln1_b.0']
# complement basis
Q,_=np.linalg.qr(np.concatenate([Th,P],1))
keys={}
for j in [0,2,3]:
    Wq=W['W_q.0'][sl(j)]; Wk=W['W_k.0'][sl(j)]
    M=Wq.T@Wk
    TT=Th.T@M@Th; TP=Th.T@M@P; PT=P.T@M@Th; PP=P.T@M@P
    # query side "other" (not T, not P) -> key T
    Oq=M.T - Q@(Q.T@M.T)   # key x other-query
    OT=(Oq.T)@Th
    print(f"L0H{j}: |TT|={np.linalg.norm(TT):.2f} |TP(q=T,k=P)|={np.linalg.norm(TP):.2f} |PT(q=P,k=T)|={np.linalg.norm(PT):.2f} |PP|={np.linalg.norm(PP):.2f} |otherq->kT|={np.linalg.norm(OT):.2f} |M|={np.linalg.norm(M):.2f}")
    s=np.linalg.svd(TT,compute_uv=False); print("    TT sv",s[:6].round(2))
    u,s,vt=np.linalg.svd(OT); print("    other->T sv",s[:4].round(2))
    keys[j]=vt[0]
    # bias based key pref
    ub=Th.T@(Wk.T@(Wq@b0)); print("    bias key pref on T norm %.2f"%np.linalg.norm(ub))
    keys[f'{j}b']=ub/np.linalg.norm(ub)
    u2,s2,vt2=np.linalg.svd(TT); keys[f'{j}tt']=vt2[0]
names=list(keys)
C=np.array([[abs(keys[a]@keys[b]) for b in names] for a in names])
print(names); print(C.round(2))
```