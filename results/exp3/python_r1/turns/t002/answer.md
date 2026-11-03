```python
import numpy as np
w=W['ln1_w.0']; print(np.percentile(w,[0,5,25,50,75,95,100]))
print(np.sort(w)[:20]); print(np.sort(w)[-20:])
w2=W['ln2_w.0']; print(np.percentile(w2,[0,5,25,50,75,95,100]))
print(np.corrcoef(w,w2)[0,1])
for l in range(4):
    print(l, np.percentile(W[f'ln1_w.{l}'],[0,5,50,95,100]).round(3), np.percentile(W[f'ln2_w.{l}'],[0,5,50,95,100]).round(3))
```