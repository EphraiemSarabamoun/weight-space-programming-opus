```python
import numpy as np
QK3=QKmat(0,3) if 'QKmat' in dir() else None
def QKmat(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][s]*wl).T@(W[f'W_k.{l}'][s]*wl)
QK3=QKmat(0,3)
ev,V=np.linalg.eig(QK3)
idx=np.argsort(-np.abs(ev))
planes=[]
used=set()
for i in idx[:40]:
    e=ev[i]
    if e.imag<=1e-6: continue
    q,_=np.linalg.qr(np.stack([V[:,i].real,V[:,i].imag],1))
    planes.append((e,q))
def rot(M,q):
    B=q.T@M@q
    ang=np.degrees(np.arctan2(B[1,0]-B[0,1], B[0,0]+B[1,1]))
    mag=np.hypot(B[1,0]-B[0,1], B[0,0]+B[1,1])/2
    return ang,mag
heads=[(0,j) for j in range(6)]+[(1,j) for j in range(6)]+[(2,j) for j in range(6)]
Ms={h:QKmat(*h) for h in heads}
for e,q in planes:
    line=f'H3eig {abs(e):.2f}@{np.degrees(np.angle(e)):5.0f} |'
    for h in heads[:6]+[(1,0),(1,3),(2,0)]:
        a,m=rot(Ms[h],q)
        line+=f' L{h[0]}H{h[1]} {a:5.0f}({m:.2f})'
    print(line)
```