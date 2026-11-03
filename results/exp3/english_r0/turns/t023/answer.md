```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
def colsp(M,k): return np.linalg.svd(M)[0][:,:k]
def orth(A): return np.linalg.qr(A)[0]
g0=g[0]
P=np.linalg.svd(np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]))[2][:20].T
B=colsp(np.hstack([OVo(0,h) for h in [0,1,4,5]]),2)
A02=colsp(OVo(0,2),20)
A03=colsp(OVo(0,3),2)
T=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:24].T
# remove P from T
T=orth(T-P@(P.T@T))[:, :24]
subs={'P':P,'B':B,'A02':A02,'A03':A03,'T':T}
names=list(subs)
for l in [1,2,3]:
    for h in range(6):
        QK=blk('W_q',l,h).T@blk('W_k',l,h)
        F=np.linalg.norm(QK)
        out=[]
        for a in names:
            for b in names:
                Sa=subs[a];Sb=subs[b]
                val=np.linalg.norm(Sa.T@QK@Sb)/F/np.sqrt(Sa.shape[1]*Sb.shape[1]/384**2)
                if val>3: out.append(f"{a}->{b}:{val:.0f}")
        # constant query via ln bias
        print(f"L{l}H{h}:", " ".join(out))
```