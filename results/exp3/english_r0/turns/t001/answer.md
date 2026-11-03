```python
import numpy as np
for k in sorted(W.keys()):
    print(k, W[k].shape, W[k].dtype, float(np.abs(W[k]).mean()))
d=384;hd=64
for l in range(4):
    print("layer",l)
    print(" ln1_w mean/std/min/max", W[f'ln1_w.{l}'].mean(), W[f'ln1_w.{l}'].std(), W[f'ln1_w.{l}'].min(), W[f'ln1_w.{l}'].max())
    print(" ln1_b mean/std", W[f'ln1_b.{l}'].mean(), W[f'ln1_b.{l}'].std(), np.abs(W[f'ln1_b.{l}']).max())
    print(" ln2_w mean/std/min/max", W[f'ln2_w.{l}'].mean(), W[f'ln2_w.{l}'].std(), W[f'ln2_w.{l}'].min(), W[f'ln2_w.{l}'].max())
    print(" ln2_b mean/std", W[f'ln2_b.{l}'].mean(), W[f'ln2_b.{l}'].std())
    for j in range(6):
        s=slice(64*j,64*j+64)
        q=W[f'W_q.{l}'][s];k=W[f'W_k.{l}'][s];v=W[f'W_v.{l}'][s];o=W[f'W_o.{l}'][:,s]
        print(f"  H{j} |q|={np.linalg.norm(q):.2f} |k|={np.linalg.norm(k):.2f} |v|={np.linalg.norm(v):.2f} |o|={np.linalg.norm(o):.2f}  sv_q top {np.linalg.svd(q,compute_uv=False)[:4].round(2)} sv_k top {np.linalg.svd(k,compute_uv=False)[:4].round(2)}")
```