```python
import numpy as np
w=W['ln1_w.0']; b=W['ln1_b.0']
print(np.histogram(w,bins=20))
print("sorted top 40", np.sort(w)[::-1][:40].round(3))
w2=W['ln2_w.0']
print(np.histogram(w2,bins=20))
print("corr ln1_w0 vs ln2_w0", np.corrcoef(w,w2)[0,1])
for l in range(1,4):
    print(l, np.histogram(W[f'ln1_w.{l}'],bins=10)[0], np.histogram(W[f'ln1_w.{l}'],bins=10)[1].round(2))
# head L0H2 QK
s=slice(128,192)
q=W['W_q.0'][s]*w; k=W['W_k.0'][s]*w
M=q.T@k
ev=np.linalg.eigvals(M)
ev=ev[np.argsort(-np.abs(ev))]
print("L0H2 eig top", ev[:20].round(2))
print("sym part norm", np.linalg.norm((M+M.T)/2), "antisym", np.linalg.norm((M-M.T)/2))
```