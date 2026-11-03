```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; M=Q.T@K
U,S,Vt=np.linalg.svd(M)
for k in [14,18,20,22]:
    X=orth(np.hstack([U[:,:k],Vt[:k].T]),k)
    R=X.T@M@X
    C=np.linalg.solve(R.T,R)
    ev=np.linalg.eigvals(C)
    ev=ev[np.argsort(np.angle(ev))]
    print(k,'cosquare |ev|:',np.round(np.abs(ev),2))
    print('   half-angles(deg):',np.round(np.degrees(np.angle(ev))/2,0))
```