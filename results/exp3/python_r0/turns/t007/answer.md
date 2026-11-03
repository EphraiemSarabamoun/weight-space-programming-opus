```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topv(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k],s
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
T,sT=topv(Vst,40); print("Vstack sv",sT[:45].round(2))
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
P,sP=topv(Pst,60); print("QK15 stack sv",sP[:60].round(2))
for kt in [24,28,32]:
  for kp in [20,30,40]:
    c=np.linalg.svd(T[:kt]@P[:kp].T,compute_uv=False)
    print(kt,kp,"cos of principal angles top", c[:6].round(2), "sum cos^2", (c**2).sum().round(2))
# L0H4 V directions and L0H0 V directions
for j in [0,4]:
    OVm=W['W_o.0'][:,sl(j)]@W['W_v.0'][sl(j)]
    u,s,vt=np.linalg.svd(OVm)
    for r in range(3):
        v=vt[r]
        print(f"L0H{j} OV right sv{r} {s[r]:.2f}: frac in T28 {np.linalg.norm(T[:28]@v)**2:.2f}, in P30 {np.linalg.norm(P[:30]@v)**2:.2f}")
```