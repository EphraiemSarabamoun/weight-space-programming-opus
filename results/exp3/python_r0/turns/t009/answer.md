```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
g0=W['ln1_w.0']; g0b=W['ln2_w.0']
A=np.concatenate([W['W_q.0'],W['W_k.0'],W['W_v.0']],0)/g0
s=np.linalg.svd(A,compute_uv=False); print("qkv0/g0", s[:90].round(1))
B=W['W_fc.0']/g0b
s=np.linalg.svd(B,compute_uv=False); print("fc0/g0b", s[:90].round(1))
# raw ratio check: average |row component| per dim vs g0
cn=np.sqrt((W['W_v.0']**2).mean(0)); print("corr colnorm Wv0 vs g0", np.corrcoef(cn,g0)[0,1])
cn=np.sqrt((W['W_q.0']**2).mean(0)); print("corr colnorm Wq0 vs g0", np.corrcoef(cn,g0)[0,1])
cn=np.sqrt((W['W_fc.0']**2).mean(0)); print("corr colnorm Wfc0 vs g0b", np.corrcoef(cn,g0b)[0,1])
# outputs of layer0 per dim
on=np.sqrt((W['W_o.0']**2).mean(1)); pn=np.sqrt((W['W_proj.0']**2).mean(1))
print("corr W_o.0 rownorm vs g0",np.corrcoef(on,g0)[0,1], "W_proj.0 rownorm vs g0", np.corrcoef(pn,g0)[0,1], "vs g0b", np.corrcoef(pn,g0b)[0,1], "W_o0 vs g0b",np.corrcoef(on,g0b)[0,1])
```