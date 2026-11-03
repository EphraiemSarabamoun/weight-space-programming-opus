```python
import numpy as np
def topk_sub(M,k,side='row'):
    U,s,Vt = np.linalg.svd(M)
    return (Vt[:k].T if side=='row' else U[:,:k]), s
def overlap(A,B):
    # mean squared cosine of principal angles
    s = np.linalg.svd(A.T@B,compute_uv=False)
    return (s**2).mean()
wl0=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    Wq=W['W_q.0'][s]*wl0; Wk=W['W_k.0'][s]*wl0; Wv=W['W_v.0'][s]*wl0; Wo=W['W_o.0'][:,s]
    QK=Wq.T@Wk
    U,S,Vt=np.linalg.svd(QK)
    k=8
    Qs=U[:,:k]; Ks=Vt[:k].T
    _,sv,Vvt=np.linalg.svd(Wv); Vs=Vvt[:k].T
    ev=np.linalg.eigvals(QK); idx=np.argsort(-np.abs(ev))[:10]
    print(f'L0H{j} QK sv',S[:10].round(2),' V sv', sv[:8].round(2))
    print('   overlap Qsub-Ksub %.2f  Qsub-Vsub %.2f Ksub-Vsub %.2f'%(overlap(Qs,Ks),overlap(Qs,Vs),overlap(Ks,Vs)))
    print('   top eig', [f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]])
    # which residual dims dominate Q and K sides
    qd=np.argsort(-np.abs(U[:,0]))[:6]; kd=np.argsort(-np.abs(Vt[0]))[:6]
    print('   top sv dims q', list(zip(qd.tolist(),U[qd,0].round(2).tolist())), ' k', list(zip(kd.tolist(),Vt[0,kd].round(2).tolist())))
    on = np.linalg.norm(Wo,axis=1); od=np.argsort(-on)[:6]
    print('   Wo rows', list(zip(od.tolist(), on[od].round(2).tolist())), 'median %.2f'%np.median(on))
    vn=np.linalg.norm(Wv,axis=0); vd=np.argsort(-vn)[:8]
    print('   Wv cols', list(zip(vd.tolist(), vn[vd].round(2).tolist())), 'median %.2f'%np.median(vn))
```