```python
import numpy as np
g0=W['ln1_w.0']
q=W['W_q.0'][128:192]*g0; k=W['W_k.0'][128:192]*g0
print("q sv",np.linalg.svd(q,compute_uv=False)[:24].round(2))
print("k sv",np.linalg.svd(k,compute_uv=False)[:24].round(2))
M=q.T@k
print("QK sv",np.linalg.svd(M,compute_uv=False)[:24].round(2))
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))][:24]
print("eig |.|,angle(deg)")
for e in ev:
    if e.imag>=0: print("  %.2f  %.1f"%(abs(e),np.degrees(np.angle(e))))
# raw (without g0)
qr=W['W_q.0'][128:192]; kr=W['W_k.0'][128:192]
print("raw q sv",np.linalg.svd(qr,compute_uv=False)[:24].round(2))
```