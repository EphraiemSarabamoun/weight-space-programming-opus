```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
V02=blk('W_v',0,2)
_,_,Vt=np.linalg.svd(V02)
OV02=W['W_o.0'][:,128:192]@V02
rng=np.random.default_rng(1)
def idl(M): return np.trace(M)/np.linalg.norm(M)
QKs={(l,h):blk('W_q',l,h).T@blk('W_k',l,h) for l in [1,2,3] for h in range(6)}
for k in [10,16,24,32]:
    T=Vt[:k].T
    out=[]
    for (l,h),QK in QKs.items():
        out.append(idl(T.T@QK@OV02@T))
    out=np.array(out).reshape(3,6)
    print("k",k,"\n",out.round(2))
# random subspace baseline for L3H1, restricted to space spanned by top 80 coords of g0
idx=np.argsort(-g[0])[:80]
vals=[]
for _ in range(200):
    R=np.zeros((384,24)); R[idx]=rng.normal(size=(80,24)); R=np.linalg.qr(R)[0]
    vals.append(idl(R.T@QKs[(3,1)]@OV02@R))
print("random-subspace baseline L3H1 mean %.2f sd %.2f"%(np.mean(vals),np.std(vals)))
```