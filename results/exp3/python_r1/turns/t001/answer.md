```python
import numpy as np
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(np.round(np.abs(a).mean(),5)), float(np.round(a.std(),5)), float(np.round(a.mean(),5)))
```