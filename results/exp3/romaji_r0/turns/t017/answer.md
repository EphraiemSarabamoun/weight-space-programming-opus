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
for i in idx[:30]:
    e=ev[i]
    if e.imag<=1e-6: continue
    q,_=np.linalg.qr(np.stack([V[:,i].real,V[:,i].imag],1))
    planes.append((e,q))
def rot(M,q):
    B=q.T@M@q
    ang=np.degrees(np.arctan2(B[1,0]-B[0,1], B[0,0]+B[1,1]))
    mag=np.hypot(B[1,0]-B[0,1], B[0,0]+B[1,1])/2
    return ang,mag
ref=[rot(QK3,q)[0] for e,q in planes]
print('H3 angles', np.round(ref))
for l in range(4):
    for j in range(6):
        M=QKmat(l,j)
        out=[rot(M,q) for e,q in planes[:8]]
        print(f'L{l}H{j}', ' '.join(f'{a:5.0f}({m:.2f})' for a,m in out))
```