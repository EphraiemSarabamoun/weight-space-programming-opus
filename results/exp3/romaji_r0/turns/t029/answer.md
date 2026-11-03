```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
# positional / L0 qk learned space
QKs=np.vstack([W['W_q.0'],W['W_k.0']])*Dl(0)
_,sq,vq=np.linalg.svd(QKs)
print('L0 qk stacked sv', sq[:80].round(2))
def run(kP,kO):
    Pq=vq[:kP].T
    Wo=W['W_o.0']; uo,so,_=np.linalg.svd(Wo); O=uo[:,:kO]
    B,_=np.linalg.qr(np.hstack([T,Pq,O]))
    F=W['W_fc.0']*W['ln2_w.0']
    Fr=F-F@B@B.T
    s=np.linalg.svd(Fr,compute_uv=False)
    # baseline: random complement directions
    return s[:15].round(2), B.shape[1]
for kP,kO in [(60,60),(60,100),(80,120)]:
    print(kP,kO, run(kP,kO))
F=W['W_fc.0']*W['ln2_w.0']
print('F sv full', np.linalg.svd(F,compute_uv=False)[:10].round(2))
```