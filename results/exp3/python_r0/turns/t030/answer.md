```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
s=np.linalg.svd(W['W_proj.3'],compute_uv=False); print("proj3",s[:40].round(2))
U3,s3,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
OVs=[W['W_o.3'][:,sl(j)]@(W['W_v.3'][sl(j)]*W['ln1_w.3']) for j in range(6)]
A=np.concatenate([W['W_proj.3']*1.0]+[o for o in OVs],1)
s=np.linalg.svd(A,compute_uv=False); print("proj3+OV3",s[:40].round(2))
for j in range(6):
    o=OVs[j]
    fr=[np.linalg.norm(U3[:,:k].T@o)**2/np.linalg.norm(o)**2 for k in (25,28,32)]
    print(f"L3H{j} OV frac in top25/28/32 of proj3:",np.round(fr,2))
s=np.linalg.svd(W['W_proj.2'],compute_uv=False); print("proj2",s[:40].round(2))
U2,_,_=np.linalg.svd(W['W_proj.2'],full_matrices=False)
print("proj2 top30 in proj3 top28:", np.linalg.norm(U3[:,:28].T@U2[:,:30])**2/30)
```