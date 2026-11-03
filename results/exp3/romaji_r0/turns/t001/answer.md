```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, 'mean %.4f std %.4f min %.3f max %.3f' % (a.mean(), a.std(), a.min(), a.max()))
d=384; hd=64
print()
for l in range(4):
    for nm in ['W_q','W_k','W_v']:
        M = W[f'{nm}.{l}']
        print(l, nm, [round(float(np.linalg.norm(M[64*j:64*j+64])),2) for j in range(6)])
    M = W[f'W_o.{l}']
    print(l, 'W_o', [round(float(np.linalg.norm(M[:,64*j:64*j+64])),2) for j in range(6)])
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        v = W[f'{nm}.{l}']
        print(l, nm, 'mean %.3f std %.3f min %.3f max %.3f' % (v.mean(), v.std(), v.min(), v.max()))
```