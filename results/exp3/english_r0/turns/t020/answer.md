```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
OV02=OVo(0,2)
rng=np.random.default_rng(0)
for l in [1,2,3]:
    row=[]
    for h in range(6):
        X=blk('W_q',l,h).T@blk('W_k',l,h)@OV02
        ev=np.linalg.eigvals(X)
        row.append(f"H{h}: tr/F {np.trace(X)/np.linalg.norm(X):+.2f} re {ev.real.sum()/np.abs(ev).sum():+.2f}")
    print(f"L{l} induction-via-L0H2:", " | ".join(row))
# control: same with other L0 heads' OV (should be ~0)
for h0 in [0,1,3,4,5]:
    O=OVo(0,h0)
    X=blk('W_q',3,1).T@blk('W_k',3,1)@O
    print("control L3H1 with L0H%d: tr/F %.2f"%(h0,np.trace(X)/np.linalg.norm(X)))
# control with random OV of same spectrum
U,S,Vt=np.linalg.svd(OV02)
for _ in range(3):
    Q1=np.linalg.qr(rng.normal(size=(384,384)))[0]; Q2=np.linalg.qr(rng.normal(size=(384,384)))[0]
    R=Q1@np.diag(S)@Q2.T
    X=blk('W_q',3,1).T@blk('W_k',3,1)@R
    print("random-OV control tr/F %.2f"%(np.trace(X)/np.linalg.norm(X)))
```