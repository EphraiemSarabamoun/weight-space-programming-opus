```python
import numpy as np
g0=W['ln1_w.0']
T=np.linalg.svd(W['W_v.0'][128:192]*g0)[2][:24].T
P=np.linalg.svd(np.vstack([W['W_q.0'][128:192]*g0,W['W_k.0'][128:192]*g0]))[2][:20].T
K=[];Qd=[]
for h in [0,1,4,5]:
    s=slice(64*h,64*h+64)
    M=(W['W_q.0'][s]*g0).T@(W['W_k.0'][s]*g0)
    u,sv,vt=np.linalg.svd(M)
    K.append(vt[0]*np.sign(vt[0]@np.ones(384)))
    print(h, "QK sv top3", sv[:3].round(3), " u.b(ln bias)=%.2f  v.b=%.2f"%(u[:,0]@W['ln1_b.0'], vt[0]@W['ln1_b.0']))
K=np.array(K)
print("stacked key dirs sv:", np.linalg.svd(K,compute_uv=False).round(2))
# project key dirs onto T and remove P; compare within T
KT=(K@T); 
KTn=KT/np.linalg.norm(KT,axis=1,keepdims=True)
print("cos between key dirs restricted to T:\n",(KTn@KTn.T).round(2))
```