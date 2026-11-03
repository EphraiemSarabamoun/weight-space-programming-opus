```python
import numpy as np
wl0=W['ln1_w.0']
j=3; s=slice(64*j,64*j+64)
Wq=W['W_q.0'][s]*wl0; Wk=W['W_k.0'][s]*wl0; Wv=W['W_v.0'][s]*wl0
print('V H3 sv', np.linalg.svd(Wv,compute_uv=False).round(3))
print('V H3 sv (no lnw)', np.linalg.svd(W['W_v.0'][s],compute_uv=False).round(3))
QK=Wq.T@Wk
ev=np.linalg.eigvals(QK); idx=np.argsort(-np.abs(ev))
print([f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx][:64]])
for j in range(6):
    s=slice(64*j,64*j+64)
    print(j,'Wv raw sv', np.linalg.svd(W['W_v.0'][s],compute_uv=False)[:40].round(2))
```