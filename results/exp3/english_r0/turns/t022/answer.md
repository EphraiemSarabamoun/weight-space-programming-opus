```python
import numpy as np
g0=W['ln1_w.0']
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g0)
Bm=np.hstack([OVo(0,h) for h in [0,1,4,5]])
U,S,Vt=np.linalg.svd(Bm); B=U[:,:3]
for h in [0,1,3,4,5]:
    O=OVo(0,h)
    Uo,So,Vo=np.linalg.svd(O)
    print(f"L0H{h} OV top2 sv {So[:2].round(2)}; out dirs in B coords:", (B.T@Uo[:,:2]).T.round(2), " in-B frac", [round(float(np.linalg.norm(B.T@Uo[:,i])**2),2) for i in range(2)])
# key-read directions (rank1 QK): angle matrix and their norm
ks={}
for h in [0,1,3,4,5]:
    s=slice(64*h,64*h+64)
    M=(W['W_q.0'][s]*g0).T@(W['W_k.0'][s]*g0)
    u,sv,vt=np.linalg.svd(M)
    ks[h]=(u[:,0],vt[0],sv[0])
H=[0,1,3,4,5]
print("key dirs |cos|:")
print(np.array([[abs(ks[a][1]@ks[b][1]) for b in H] for a in H]).round(2))
print("query dirs |cos|:")
print(np.array([[abs(ks[a][0]@ks[b][0]) for b in H] for a in H]).round(2))
# key dirs of broad heads vs L0H2 V read space (token copying) top-30
T=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:30].T
for h in H: print(h,"key dir frac in T30: %.2f  query dir frac in T30: %.2f"%(np.linalg.norm(T.T@ks[h][1])**2,np.linalg.norm(T.T@ks[h][0])**2))
```