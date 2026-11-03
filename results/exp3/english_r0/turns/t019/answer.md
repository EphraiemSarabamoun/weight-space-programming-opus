```python
import numpy as np
g0=W['ln1_w.0']
q=W['W_q.0'][128:192]*g0; k=W['W_k.0'][128:192]*g0; v=W['W_v.0'][128:192]*g0
M=q.T@k
ev,EV=np.linalg.eig(M)
order=np.argsort(-np.abs(ev))
pairs=[]
for i in order[:40]:
    e=ev[i]
    if e.imag<-1e-6: continue
    vv=EV[:,i]
    S=np.linalg.qr(np.stack([vv.real,vv.imag],1))[0] if abs(e.imag)>1e-6 else (vv.real/np.linalg.norm(vv.real))[:,None]
    pairs.append((abs(e),np.degrees(np.angle(e)),S))
print("n subspaces",len(pairs))
Vr=np.linalg.svd(v)[2][:24].T
for m,a,S in pairs[:14]:
    # how much of L0H2 V reads it, and the broad heads V
    fv=np.linalg.norm(v@S)**2/np.linalg.norm(v)**2/(S.shape[1]/384)
    fb=[np.linalg.norm((W['W_v.0'][64*h:64*h+64]*g0)@S)**2/np.linalg.norm(W['W_v.0'][64*h:64*h+64]*g0)**2/(S.shape[1]/384) for h in [0,1,3,4,5]]
    ffc=np.linalg.norm((W['W_fc.0']*W['ln2_w.0'])@S)**2/np.linalg.norm(W['W_fc.0']*W['ln2_w.0'])**2/(S.shape[1]/384)
    print(f"|ev|={m:.2f} ang={a:6.1f} dim={S.shape[1]}  L0H2 v-read x{fv:.1f}  broad v-read {np.round(fb,1)}  fc0 x{ffc:.1f}")
```