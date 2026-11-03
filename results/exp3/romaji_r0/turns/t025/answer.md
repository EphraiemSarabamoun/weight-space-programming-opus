```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
F=W['W_fc.0']*W['ln2_w.0']
Pr=W['W_proj.0']
ns=np.argsort(-Pr[184])[:8]
Fn=F[ns]/np.linalg.norm(F[ns],axis=1,keepdims=True)
m=Fn.mean(0); m/=np.linalg.norm(m)
St=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])*Dl(0)
_,s,vt=np.linalg.svd(St)
for k in [20,40,60,100,150]:
    print('E top',k,'energy of m %.3f (rand %.3f)'%(np.linalg.norm(vt[:k]@m)**2,k/384))
u,so,_=np.linalg.svd(W['W_o.0'])
for k in [5,20,60,100]:
    print('Wo0 out top',k,'energy of m %.3f'%(np.linalg.norm(u[:,:k].T@m)**2))
print('Wo0 top sv', so[:8].round(2))
print('cos(m, Wo0 top 5 out dirs)', (u[:,:5].T@m).round(2))
# what about ln2_b.0 direction
b=W['ln2_b.0']; print('neurons bias input F@b for these', (W['W_fc.0'][ns]@b).round(2), ' typical |F@b| median', np.median(np.abs(W['W_fc.0']@b)).round(2))
```