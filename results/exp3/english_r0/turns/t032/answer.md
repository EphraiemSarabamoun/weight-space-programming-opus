```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
def colsp(M,k): return np.linalg.svd(M)[0][:,:k]
g0=g[0]
subs={}
subs['P']=np.linalg.svd(np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]))[2][:20].T
subs['T']=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:24].T
subs['A02']=colsp(OVo(0,2),20)
subs['B']=colsp(np.hstack([OVo(0,h) for h in [0,1,4,5]]),2)
subs['M0']=colsp(W['W_proj.0'],30)
for h in range(6): subs[f'A1.{h}']=colsp(OVo(1,h),15)
subs['M1']=colsp(W['W_proj.1'],30)
for h in range(6): subs[f'A2.{h}']=colsp(OVo(2,h),15)
subs['M2']=colsp(W['W_proj.2'],30)
for (l,h) in [(3,0),(3,1),(3,5),(2,1),(2,2)]:
    QK=blk('W_q',l,h).T@blk('W_k',l,h)
    out=[]
    for n,S in subs.items():
        Mq=S.T@QK@S
        idl=np.trace(Mq)/np.linalg.norm(QK)  # trace within subspace relative to total
        out.append(f"{n}:{idl:+.2f}")
    print(f"L{l}H{h} trace-within-subspace/|QK|:", " ".join(out))
```