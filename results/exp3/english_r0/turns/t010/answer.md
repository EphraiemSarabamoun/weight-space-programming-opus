```python
import numpy as np
g0=W['ln1_w.0']
def blk(nm,h): return W[nm][64*h:64*h+64]*g0
def rowspace(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Pq=rowspace(blk('W_q.0',2),12); Pk=rowspace(blk('W_k.0',2),12); Pv=rowspace(blk('W_v.0',2),20)
print("sv v L0H2:",np.linalg.svd(blk('W_v.0',2),compute_uv=False)[:30].round(2))
print("sv q L0H2:",np.linalg.svd(blk('W_q.0',2),compute_uv=False)[:20].round(2))
print("sv k L0H2:",np.linalg.svd(blk('W_k.0',2),compute_uv=False)[:20].round(2))
def overlap(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
print("q-k principal cosines",overlap(Pq,Pk).round(2))
print("q-v principal cosines",overlap(Pq,Pv).round(2))
print("k-v principal cosines",overlap(Pk,Pv).round(2))
```