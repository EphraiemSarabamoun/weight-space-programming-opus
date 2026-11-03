```python
import numpy as np
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(a.mean()), float(a.std()), float(np.abs(a).max()))
```