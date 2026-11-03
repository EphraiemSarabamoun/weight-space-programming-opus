```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,30)
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
B={j:P.T@(W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)])@P for j in range(6)}
n1=np.linalg.norm(B[1]); n5=np.linalg.norm(B[5])
Y=0.6*B[1]/n1+0.8*B[5]/n5
ev,Wv=np.linalg.eig(Y)
planes=[]
for k in range(len(ev)):
    if ev[k].imag< -1e-8: continue
    w=Wv[:,k]
    if abs(ev[k].imag)>1e-8:
        Q,_=np.linalg.qr(np.stack([w.real,w.imag],1))
    else:
        Q=(w.real/np.linalg.norm(w.real))[:,None]
    lam5=np.vdot(w/np.linalg.norm(w),B[5]@(w/np.linalg.norm(w)))
    planes.append((np.degrees(np.angle(lam5)),abs(lam5),P@Q))
planes.sort(key=lambda t:abs(t[0]))
print("planes (omega from H5):",[f"{a:.0f}({m:.1f})" for a,m,_ in planes])
# readers: L0H0 query, L0H4 V, L0H0 V, L0H2/3 q,k
def frac(Mat,Q): return np.linalg.norm(Mat@Q)**2
for name,Mat in [('H0q',W['W_q.0'][sl(0)]),('H0k',W['W_k.0'][sl(0)]),('H0v',W['W_v.0'][sl(0)]),('H4v',W['W_v.0'][sl(4)]),('H2q',W['W_q.0'][sl(2)]),('H2k',W['W_k.0'][sl(2)]),('H3q',W['W_q.0'][sl(3)]),('H3k',W['W_k.0'][sl(3)])]:
    tot=np.linalg.norm(Mat)**2
    print(name, "tot %.1f"%tot, " ".join(f"{a:.0f}:{frac(Mat,Q):.2f}" for a,m,Q in planes), " T:%.2f"%frac(Mat,Th))
```