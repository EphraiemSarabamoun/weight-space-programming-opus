```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k=None):
    U,S,_=np.linalg.svd(M,full_matrices=False); 
    if k is None: k=(S>S[0]*1e-3).sum()
    return U[:,:k]
Q,K,V,O=H[(0,2)]
U,S,Vt=np.linalg.svd(Q.T@K)
P=orth(np.hstack([U[:,:26],Vt[:26].T]),30)
Uv,Sv,Vv=np.linalg.svd(O@V); T=orth(Vv[:24].T-P@(P.T@Vv[:24].T),20)
B=orth(np.hstack([H[(0,h)][3]@H[(0,h)][2] for h in [0,1,4,5]]),3)
P2=orth(O@V,20)
Um=orth(W['W_proj.0'],None); 
sm=np.linalg.svd(W['W_proj.0'],compute_uv=False); print('W_proj.0 sv',np.round(sm[:30],2))
M0=orth(W['W_proj.0'],40)
subs={'P':P,'T':T,'B':B,'P2':P2,'M0':M0}
for nm,A in subs.items():
    print(nm,{n2:np.round(np.linalg.svd(A.T@B2,compute_uv=False)[:3],2).tolist() for n2,B2 in subs.items() if n2!=nm})
def fr(M,B): return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
for l in [1,2,3]:
    for h in range(6):
        Q,K,V,O=H[(l,h)]
        print(l,h,' '.join('%s:q%.2f k%.2f v%.2f'%(nm,fr(Q,A),fr(K,A),fr(V,A)) for nm,A in subs.items()))
```