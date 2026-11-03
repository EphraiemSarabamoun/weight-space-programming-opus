```python
import numpy as np
def insp(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
g0=W['ln1_w.0']; g2=W['ln2_w.0']
Qk=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
P=insp(Qk,20); Pr=np.eye(384)-P@P.T
Traw=insp(np.vstack([W['W_v.0'][64:128],W['W_v.0'][320:384]])@Pr,27)
for nm,E in [('div',np.linalg.qr(Traw/g0[:,None])[0]),('raw',Traw)]:
    F=W['W_fc.0']*g2
    A=F@E   # 1536x27
    nrm=np.linalg.norm(A,axis=1); full=np.linalg.norm(F,axis=1)
    print(nm, "frac of neuron norm^2 in E: median", np.median((nrm/full)**2), "p90", np.percentile((nrm/full)**2,90))
    sv=np.linalg.svd(A,compute_uv=False); print("  sv of A", np.round(sv,1))
    # sparsity: for each neuron, kurtosis-like measure of its coordinates in a varimax rotation? first just check coordinate distributions
```