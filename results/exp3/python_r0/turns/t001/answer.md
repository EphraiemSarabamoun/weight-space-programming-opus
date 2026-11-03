```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    print(k, W[k].shape, W[k].dtype, float(np.abs(W[k]).mean()), float(W[k].std()))
for l in range(4):
    print("layer", l)
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        a = W[f'{nm}.{l}']
        print(nm, 'mean %.3f std %.3f min %.3f max %.3f' % (a.mean(), a.std(), a.min(), a.max()))
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        s = np.linalg.svd(W[f'{nm}.{l}'], compute_uv=False)
        print(nm, s[:40].round(2), '... median', np.median(s).round(3), 'min', s.min().round(3))
```