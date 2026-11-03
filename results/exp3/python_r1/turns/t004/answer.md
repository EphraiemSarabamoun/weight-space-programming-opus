```python
import numpy as np
g=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    Wq=W['W_q.0'][s]*g; Wk=W['W_k.0'][s]*g
    ev=np.linalg.eigvals(Wk@Wq.T)
    o=np.argsort(-np.abs(ev))[:16]
    print(f"L0H{j}", " ".join(f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e in ev[o]))
    sv=np.linalg.svd(Wq.T@Wk,compute_uv=False)
    print("   sv", np.round(sv[:12],2))
```