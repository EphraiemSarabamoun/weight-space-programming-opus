```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
Ms=[H[0][j][0].T@H[0][j][1] for j in range(6)]
def planes(M,n):
    ev,evec=np.linalg.eig(M)
    o=np.argsort(-np.abs(ev)); out=[]; used=set()
    for t in o:
        if len(out)>=n: break
        if ev[t].imag>1e-6:
            v=evec[:,t]; Q,_=np.linalg.qr(np.stack([v.real,v.imag],1)); out.append((ev[t],Q))
    return out
def rot(M2):
    a=(M2[0,0]+M2[1,1])/2; b=(M2[1,0]-M2[0,1])/2
    return np.degrees(np.arctan2(b,a)), np.hypot(a,b)
for ref in [1,5]:
    print("reference planes of L0H%d"%ref)
    for ev,Q in planes(Ms[ref],10):
        s=f"  ev {abs(ev):.2f}@{np.degrees(np.angle(ev)):.0f}: "
        for j in range(6):
            ang,mag=rot(Q.T@Ms[j]@Q)
            s+=f"H{j} {ang:+.0f}/{mag:.2f}  "
        print(s)
```