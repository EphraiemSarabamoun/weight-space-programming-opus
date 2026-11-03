```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def QKmat(l,j):
    s=slice(64*j,64*j+64); wl=Dl(l)
    return (W[f'W_q.{l}'][s]*wl).T@(W[f'W_k.{l}'][s]*wl)
QK3=QKmat(0,3)
ev,V=np.linalg.eig(QK3)
idx=np.argsort(-np.abs(ev))
planes=[]
for i in idx[:34]:
    e=ev[i]
    if e.imag<=1e-6: continue
    q,_=np.linalg.qr(np.stack([V[:,i].real,V[:,i].imag],1))
    planes.append((e,q))
def rot(M,q):
    B=q.T@M@q
    return np.arctan2(B[1,0]-B[0,1], B[0,0]+B[1,1]), np.hypot(B[1,0]-B[0,1], B[0,0]+B[1,1])/2
ref=[rot(QK3,q) for e,q in planes]
D=np.arange(0,25)
def kernel(M, weight='eq'):
    K=np.zeros(len(D))
    for (e,q),(a3,m3) in zip(planes,ref):
        s=np.sign(a3); om=abs(a3)
        a,m=rot(M,q); phi=s*a
        w = 1.0 if weight=='eq' else abs(e)
        K+= w*m*np.cos(om*D-phi)
    return K
print('planes used', len(planes), 'omegas(deg)', np.round([np.degrees(abs(r[0])) for r in ref]))
for l in range(4):
    for j in range(6):
        K=kernel(QKmat(l,j))
        print(f'L{l}H{j} argmax {D[np.argmax(K)]:2d}', np.round(K[:16],2))
```