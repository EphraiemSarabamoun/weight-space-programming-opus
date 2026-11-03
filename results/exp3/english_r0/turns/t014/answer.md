```python
import numpy as np
g0=W['ln1_w.0']
A=np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]); P=np.linalg.svd(A)[2][:12].T
T=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:24].T
eP=(P**2).sum(1); eT=(T**2).sum(1)
order=np.argsort(-g0)
print("top coords by ln1_w0: w, P-energy*384/12, T-energy*384/24")
for i in order[:60]:
    print(i, round(float(g0[i]),2), round(float(eP[i]*384/12),2), round(float(eT[i]*384/24),2))
print("corr g0 vs eP", np.corrcoef(g0,eP)[0,1], " g0 vs eT", np.corrcoef(g0,eT)[0,1])
print("raw Wv02 sv", np.linalg.svd(W['W_v.0'][128:192],compute_uv=False).round(2))
```