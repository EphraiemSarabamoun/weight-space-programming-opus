```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
V02=blk('W_v',0,2)
U,S,Vt=np.linalg.svd(V02); T=Vt[:24].T
OV02=W['W_o.0'][:,128:192]@V02
rng=np.random.default_rng(1)
def idl(M):  # identity-likeness
    return np.trace(M)/np.linalg.norm(M)*1.0
for l in [1,2,3]:
    row=[]
    for h in range(6):
        QK=blk('W_q',l,h).T@blk('W_k',l,h)
        Mind=T.T@QK@OV02@T     # query current tok, key prev tok (from L0H2)
        Msame=T.T@QK@T
        Mrev=T.T@OV02.T@QK@T   # query prev-token feature vs key current token
        row.append(f"H{h}: ind {idl(Mind):+.2f} same {idl(Msame):+.2f} prevq {idl(Mrev):+.2f}")
    print(f"L{l}", " | ".join(row))
# random baseline: random 24-dim subspace
Tr=np.linalg.qr(rng.normal(size=(384,24)))[0]
print("baseline sd ~", 1.0)
```