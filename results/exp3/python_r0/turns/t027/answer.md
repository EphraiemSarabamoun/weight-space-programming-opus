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
Y=0.6*B[1]/n1+0.8*B[5]/n5
ev,Wv=np.linalg.eig(Y)
planes=[]
for k in range(len(ev)):
    if ev[k].imag< -1e-8: continue
    w=Wv[:,k]/np.linalg.norm(Wv[:,k])
    lam={h:np.vdot(w,B[h]@w) for h in range(6)}
    mult=2 if abs(ev[k].imag)>1e-8 else 1
    planes.append((lam,mult))
D=np.arange(0,13)
for ref in [5,1]:
  print("reference head (assumed Delta=1):",ref)
  for h in [1,5,2,3]:
    prof=np.zeros(len(D))
    for lam,mult in planes:
        om=np.angle(lam[ref])
        prof+=mult*np.abs(lam[h])*np.cos(np.angle(lam[h])-om*D)
    print(f"  head {h} profile over Delta 0..12:", prof.round(1), "argmax", D[np.argmax(prof)])
```