```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,30)
for heads in [[1,5],[1,2,3,5],[0,1,2,3,4,5],[2,3]]:
    Vst=np.concatenate([W['W_v.0'][sl(j)] for j in heads],0)
    s=np.linalg.svd(Vst,compute_uv=False)
    Vp=Vst-(Vst@Ph)@Ph.T
    s2=np.linalg.svd(Vp,compute_uv=False)
    print(heads,"raw",s[:34].round(2))
    print(heads,"noP",s2[:34].round(2))
# Also the K/Q of content heads 2,3 projected out of P
for j in [2,3]:
    for kind in 'qk':
        M=W[f'W_{kind}.0'][sl(j)]; Mp=M-(M@Ph)@Ph.T
        print(j,kind,np.linalg.svd(Mp,compute_uv=False)[:32].round(2))
```