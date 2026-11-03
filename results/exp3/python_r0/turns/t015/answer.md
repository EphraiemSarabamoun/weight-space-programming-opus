```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,_=topV(Vst,27)
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,sp=topV(Pst,40)
All=np.concatenate([W['W_q.0'],W['W_k.0'],W['W_v.0'],W['W_fc.0']],0)
Ah,sa=topV(All,120)
for j in [0,2,3,4]:
    OVm=W['W_o.0'][:,sl(j)]@W['W_v.0'][sl(j)]
    u,s,vt=np.linalg.svd(OVm)
    for r in range(2):
        v=vt[r]
        out=[np.linalg.norm(Th.T@v)**2]+[np.linalg.norm(Ph[:,:k].T@v)**2 for k in (10,20,30,40)]+[np.linalg.norm(Ah[:,:k].T@v)**2 for k in (40,80,120)]
        # project out T then P20
        Q,_=np.linalg.qr(np.concatenate([Th,Ph[:,:20]],1))
        print(f"L0H{j} sv{r} {s[r]:.2f}  T27 {out[0]:.2f} P10 {out[1]:.2f} P20 {out[2]:.2f} P30 {out[3]:.2f} P40 {out[4]:.2f} | All40 {out[5]:.2f} All80 {out[6]:.2f} All120 {out[7]:.2f} | T+P20 {np.linalg.norm(Q.T@v)**2:.2f}")
# Also: Q and K of L0H0 and L0H4 directions
for j in [0,2,3,4]:
    for kind in 'qk':
        M=W[f'W_{kind}.0'][sl(j)]
        u,s,vt=np.linalg.svd(M)
        v=vt[0]
        print(f"L0H{j}{kind} top dir sv {s[0]:.2f}: T27 {np.linalg.norm(Th.T@v)**2:.2f} P20 {np.linalg.norm(Ph[:,:20].T@v)**2:.2f}")
```