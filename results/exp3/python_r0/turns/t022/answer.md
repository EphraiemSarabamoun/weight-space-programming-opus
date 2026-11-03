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
B={}
for j in [1,5]:
    M=W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)]
    B[j]=P.T@M@P
def comm(X,Y): return np.linalg.norm(X@Y-Y@X)/(np.linalg.norm(X)*np.linalg.norm(Y))
S1=(B[1]+B[1].T)/2;A1=(B[1]-B[1].T)/2;S5=(B[5]+B[5].T)/2;A5=(B[5]-B[5].T)/2
rng=np.random.default_rng(0)
R1=rng.standard_normal((24,24));R2=rng.standard_normal((24,24))
print("commutators: S1S5 %.2f A1A5 %.2f S1A1 %.2f S5A5 %.2f S1A5 %.2f | random sym-sym %.2f"%(comm(S1,S5),comm(A1,A5),comm(S1,A1),comm(S5,A5),comm(S1,A5),comm(R1+R1.T,R2+R2.T)))
# eigen planes of A5 (strongest antisym)
for name,X in [('A5',A5),('A1',A1),('A1+A5',A1/np.linalg.norm(A1)+0.7*A5/np.linalg.norm(A5))]:
    ev,V=np.linalg.eig(X)
    o=np.argsort(-ev.imag); ev=ev[o]; V=V[:,o]
    print("planes from",name)
    for k in range(10):
        if ev[k].imag<=1e-6: break
        v=V[:,k]; Q,_=np.linalg.qr(np.stack([v.real,v.imag],1))
        res=[]
        for h,(S,A) in {1:(S1,A1),5:(S5,A5)}.items():
            Bp=Q.T@B[h]@Q
            alpha=(Bp[0,0]+Bp[1,1])/2; beta=(Bp[1,0]-Bp[0,1])/2
            # invariance: fraction of B Q energy staying in plane
            inv=np.linalg.norm(Q.T@B[h]@Q)/np.linalg.norm(B[h]@Q)
            res.append(f"H{h}: r={np.hypot(alpha,beta):5.2f} th={np.degrees(np.arctan2(beta,alpha)):6.0f} inv={inv:.2f} aniso={abs(Bp[0,0]-Bp[1,1])/2:.2f}")
        print("  ", " | ".join(res))
```