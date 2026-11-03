```python
import numpy as np
w=W['ln1_w.0']
hi=w>0.5
print("n hi",hi.sum())
for j in range(6):
    s=slice(64*j,64*j+64)
    for name in ['W_q.0','W_k.0','W_v.0']:
        A=W[name][s]
        cn=np.linalg.norm(A,axis=0)
        cnw=np.linalg.norm(A*w,axis=0)
        print(j,name,"raw colnorm hi/lo mean %.3f %.3f"%(cn[hi].mean(),cn[~hi].mean()), " effective (x ln_w) energy frac in hi: %.3f"%((cnw[hi]**2).sum()/(cnw**2).sum()))
# fc
A=W['W_fc.0']; w2=W['ln2_w.0']
cn=np.linalg.norm(A,axis=0)
print("fc0 colnorm hi/lo", cn[hi].mean(), cn[~hi].mean())
print("ln2_w0 hi/lo", w2[hi].mean(), w2[~hi].mean())
print("W_o.0 rownorm hi/lo", np.linalg.norm(W['W_o.0'],axis=1)[hi].mean(), np.linalg.norm(W['W_o.0'],axis=1)[~hi].mean())
print("W_proj.0 rownorm hi/lo", np.linalg.norm(W['W_proj.0'],axis=1)[hi].mean(), np.linalg.norm(W['W_proj.0'],axis=1)[~hi].mean())
for l in range(1,4):
    print(l,"ln1_w hi/lo", W[f'ln1_w.{l}'][hi].mean(), W[f'ln1_w.{l}'][~hi].mean(), "ln1_b abs hi/lo", np.abs(W[f'ln1_b.{l}'][hi]).mean(), np.abs(W[f'ln1_b.{l}'][~hi]).mean())
```