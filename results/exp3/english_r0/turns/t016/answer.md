```python
import numpy as np
g={l:W[f'ln1_w.{l}'] for l in range(4)}
def blk(nm,l,h): return W[f'{nm}.{l}'][64*h:64*h+64]*g[l]
def OVo(l,h): s=slice(64*h,64*h+64); return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g[l])
def colsp(M,k): U,S,Vt=np.linalg.svd(M); return U[:,:k]
subs={}
g0=g[0]
A=np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]); subs['P(L0H2qk)']=np.linalg.svd(A)[2][:12].T
subs['T(L0H2v)']=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:20].T
subs['A0.2out']=colsp(OVo(0,2),20)
subs['B']=colsp(np.hstack([OVo(0,h) for h in [0,1,4,5]]),3)
subs['M0out']=colsp(W['W_proj.0'],40)
subs['L1out']=colsp(np.hstack([OVo(1,h) for h in range(6)]),40)
subs['M1out']=colsp(W['W_proj.1'],40)
subs['L2out']=colsp(np.hstack([OVo(2,h) for h in range(6)]),40)
subs['M2out']=colsp(W['W_proj.2'],40)
for (l,h) in [(3,0),(3,1),(3,5),(2,1)]:
    Q=blk('W_q',l,h); K=blk('W_k',l,h)
    uq=np.linalg.svd(Q)[2][:16].T; uk=np.linalg.svd(K)[2][:16].T
    print(f"L{l}H{h}: q-read(16) energy in subspace / random expectation; k-read likewise")
    for n,S in subs.items():
        k=S.shape[1]
        fq=(np.linalg.norm(Q@S)**2/np.linalg.norm(Q)**2)/(k/384)
        fk=(np.linalg.norm(K@S)**2/np.linalg.norm(K)**2)/(k/384)
        print(f"   {n:10s} q {fq:5.2f}  k {fk:5.2f}")
```