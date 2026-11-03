```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
g0=g[0]
q=blk('W_q',0,2); k=blk('W_k',0,2); M=q.T@k
ev,EV=np.linalg.eig(M); i=np.argmax(np.abs(ev)); 
if ev[i].imag<0: i=[j for j in range(len(ev)) if abs(ev[j]-np.conj(ev[i]))<1e-6][0]
v=EV[:,i]; S=np.linalg.qr(np.stack([v.real,v.imag],1))[0]
print("L0H2 eig",ev[i], "block in plane", (S.T@M@S).round(2))
U,Sv,Vt=np.linalg.svd(np.hstack([OVo(0,h) for h in [0,1,4,5]])); B=U[:,:2]
Os={}
for h in [0,1,4,5,3]:
    O=B.T@OVo(0,h)@S
    Os[h]=O
    sv=np.linalg.svd(O,compute_uv=False)
    print(f"L0H{h} plane->B map", O.round(2).tolist(), "sv",sv.round(2), "det sign", np.sign(np.linalg.det(O)))
O=sum(Os[h] for h in [0,1,4,5])
for l,h in [(1,2),(2,0),(3,3)]:
    QK=blk('W_q',l,h).T@blk('W_k',l,h)
    Mb=B.T@QK@B
    for hh in [0,5]:
        Mp=Os[hh].T@Mb@Os[hh]
        e=np.linalg.eigvals(Mp)
        print(f"L{l}H{h} B->B in plane coords via L0H{hh}:", Mp.round(2).tolist(), "eig angle", np.degrees(np.angle(e)).round(1))
    # P->B: raw query plane vs key B
    Mpb=S.T@QK@B@Os[5]
    print("  query plane -> key B(plane coords via H5):", Mpb.round(2).tolist(), np.degrees(np.angle(np.linalg.eigvals(Mpb))).round(1))
```