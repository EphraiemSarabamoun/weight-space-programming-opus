```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
Q,K,V,O=H[(0,2)]
U,S,Vt=np.linalg.svd(Q.T@K)
P=np.linalg.svd(np.hstack([U[:,:26],Vt[:26].T]))[0][:,:30]  # positional subspace approx
Uv,Sv,Vv=np.linalg.svd(O@V); T0=Vv[:24].T
T=T0-P@(P.T@T0); T=np.linalg.svd(T)[0][:,:20]
print('T sv after orth', np.round(np.linalg.svd(T0-P@(P.T@T0),compute_uv=False),2))
for h in range(6):
    Q,K,V,O=H[(0,h)]
    def fr(M,B): return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
    print(h,'q P %.2f T %.2f | k P %.2f T %.2f | v P %.2f T %.2f'%(fr(Q,P),fr(Q,T),fr(K,P),fr(K,T),fr(V,P),fr(V,T)))
    M=Q.T@K
    for nm,A,Bm in [('PP',P,P),('PT',P,T),('TP',T,P),('TT',T,T)]:
        print('   ',nm,'%.2f'%np.linalg.norm(A.T@M@Bm),end='')
    print(' tot %.2f'%np.linalg.norm(M))
    OV=O@V
    print('    OV: from P %.2f from T %.2f tot %.2f'%(np.linalg.norm(OV@P),np.linalg.norm(OV@T),np.linalg.norm(OV)))
```