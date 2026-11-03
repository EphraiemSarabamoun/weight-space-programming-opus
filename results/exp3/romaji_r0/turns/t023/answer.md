```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
F=W['W_fc.0']*W['ln2_w.0']
Pr=W['W_proj.0']
ns=np.argsort(-Pr[184])[:12]
print('top 184 writers', ns, Pr[184,ns].round(2), 'dim21', Pr[21,ns].round(2))
Fn=F[ns]/np.linalg.norm(F[ns],axis=1,keepdims=True)
print('cos sim among their input rows\n', (Fn@Fn.T).round(2))
m=Fn.mean(0); m/=np.linalg.norm(m)
idx=np.argsort(-np.abs(m))[:15]
print('mean input dir top dims', list(zip(idx.tolist(), m[idx].round(2).tolist())))
# compare with L0 head subspaces
for j in range(6):
    s=slice(64*j,64*j+64)
    for nm in ['W_q','W_k','W_v']:
        M=W[f'{nm}.0'][s]*Dl(0)
        _,sv,vt=np.linalg.svd(M)
        print(j,nm,'proj of m on top-3 dirs', np.round(vt[:3]@m,2))
    Wo=W['W_o.0'][:,s]; u,sv,_=np.linalg.svd(Wo@ (W['W_v.0'][s]*Dl(0)))
    print(j,'OV out top-3 dirs cos with m', np.round(u[:,:3].T@m,2))
```