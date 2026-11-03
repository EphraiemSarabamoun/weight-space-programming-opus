```python
import numpy as np
g0=W['ln1_w.0']
q=W['W_q.0'][128:192]*g0; k=W['W_k.0'][128:192]*g0
M=q.T@k
ev,EV=np.linalg.eig(M)
order=np.argsort(-np.abs(ev))[:40]
# build real subspaces for each eigen (pair)
subsp=[]
seen=set()
for i in order:
    e=ev[i]
    if e.imag< -1e-6: continue
    v=EV[:,i]
    if abs(e.imag)>1e-6:
        S=np.linalg.qr(np.stack([v.real,v.imag],1))[0]
    else:
        S=(v.real/np.linalg.norm(v.real))[:,None]
    subsp.append((abs(e),np.degrees(np.angle(e)),S))
def vread(h,nm='W_v.0',k=2):
    A=W[nm][64*h:64*h+64]*g0; return np.linalg.svd(A)[2][:k]
for h in [0,1,3,4,5]:
    for nm in ['W_v.0','W_q.0','W_k.0']:
        dirs=vread(h,nm,1)
        d=dirs[0]
        proj=[(round(m,1),round(a),round(float(np.linalg.norm(S.T@d)**2),2)) for m,a,S in subsp[:16]]
        print(f"L0H{h} {nm} top-read dir: ", [p for p in proj if p[2]>0.04])
```