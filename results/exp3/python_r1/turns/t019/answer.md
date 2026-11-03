```python
import numpy as np
A=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])
sv=np.linalg.svd(A,compute_uv=False)
print(np.round(sv,3))
```