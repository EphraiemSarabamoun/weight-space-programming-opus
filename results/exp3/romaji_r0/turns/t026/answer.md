```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
# outputs of L0H3 (prev token) space
s3=slice(192,256)
OV3=W['W_o.0'][:,s3]@blk('W_v',0,3)
u3,_,_=np.linalg.svd(OV3@T); O3=u3[:,:24]   # where prev-token info (of tokens) is written
Sd=[21,39,58,99,184,383]
for l in range(1,4):
    for j in range(6):
        res=[]
        for nm in ['W_q','W_k','W_v']:
            M=blk(nm,l,j); tot=np.linalg.norm(M)**2
            res.append(f'{nm[-1]}: T {np.linalg.norm(M@T)**2/tot*384/24:4.1f} O3 {np.linalg.norm(M@O3)**2/tot*384/24:4.1f} sink {np.linalg.norm(M[:,Sd])**2/tot*384/6:5.1f}')
        print(f'L{l}H{j}', ' | '.join(res))
```