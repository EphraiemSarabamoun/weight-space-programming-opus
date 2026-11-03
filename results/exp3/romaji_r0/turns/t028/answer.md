```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
u,s,vt=np.linalg.svd(A)
def eng(M):
    Mn=np.linalg.norm(M)**2
    return np.array([np.linalg.norm(M@vt[k])**2/Mn*384 for k in range(128)])
F=eng(W['W_fc.0']*W['ln2_w.0'])
Q=eng(np.vstack([W['W_q.0'],W['W_k.0']])*Dl(0))
V1=eng(W['W_v.1']*Dl(1)); Q1=eng(W['W_q.1']*Dl(1)); F1=eng(W['W_fc.1']*W['ln2_w.1'])
for k in range(128):
    flag = '*' if (F[k]>1.5 and Q[k]<1.0) else ''
    print(k, f'sv {s[k]:.2f} MLP0 {F[k]:.1f} L0qk {Q[k]:.1f} L1v {V1[k]:.1f} L1q {Q1[k]:.1f} L1fc {F1[k]:.1f}', flag)
```