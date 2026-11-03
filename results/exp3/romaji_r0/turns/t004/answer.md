```python
import numpy as np
for l in range(4):
    ro = np.linalg.norm(W[f'W_o.{l}'],axis=1); rp = np.linalg.norm(W[f'W_proj.{l}'],axis=1)
    cq = np.linalg.norm(W[f'W_q.{l}']*W[f'ln1_w.{l}'],axis=0); ck = np.linalg.norm(W[f'W_k.{l}']*W[f'ln1_w.{l}'],axis=0)
    cf = np.linalg.norm(W[f'W_fc.{l}']*W[f'ln2_w.{l}'],axis=0)
    print('layer',l)
    for nm,v in [('Wo rows',ro),('Wproj rows',rp),('Wq cols(lnw)',cq),('Wk cols(lnw)',ck),('Wfc cols',cf)]:
        idx = np.argsort(-v)[:8]
        print('  ',nm, 'median %.3f'%np.median(v), list(zip(idx.tolist(), v[idx].round(2).tolist())))
    for nm in ['ln1_b','ln2_b','ln1_w','ln2_w']:
        v=W[f'{nm}.{l}']; idx=np.argsort(-np.abs(v))[:8]
        print('  ',nm, list(zip(idx.tolist(), v[idx].round(2).tolist())))
```