```python
import numpy as np
g=W['ln1_w.0']; g2=W['ln2_w.0']
print("count ln1w0>0.5:", (g>0.5).sum(), ">0.3:", (g>0.3).sum(), ">0.7", (g>0.7).sum())
idx=np.argsort(-g)
print(np.round(g[idx[:60]],2))
# for each head, fraction of (raw, unscaled) Wq/Wk/Wv column norm^2 in the top-k ln dims
top=idx[:40]
for j in range(6):
    s=slice(64*j,64*j+64)
    for nm in ['W_q.0','W_k.0','W_v.0']:
        A=W[nm][s]*g
        f=(A[:,top]**2).sum()/(A**2).sum()
        print(f"L0H{j} {nm} frac in top40 g dims {f:.2f}", end=' | ')
    print()
F=W['W_fc.0']*g2
print("fc0 frac top40", (F[:,top]**2).sum()/(F**2).sum())
print("ln2w0 at top40 dims", np.round(g2[top],2))
```