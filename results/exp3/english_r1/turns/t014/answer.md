```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]
U,S,Vt=np.linalg.svd(Q.T@K)
P=orth(np.hstack([U[:,:26],Vt[:26].T]),30)
Uv,Sv,Vv=np.linalg.svd(O@V); T=orth(Vv[:24].T-P@(P.T@Vv[:24].T),20)
w=W['ln1_w.0']
pc=(P**2).sum(1); tc=(T**2).sum(1)
order=np.argsort(-w)
print('coord: w, Pmass, Tmass for top-w coords')
for i in order[:50]: print(i, round(float(w[i]),2), round(float(pc[i]),3), round(float(tc[i]),3))
print('corr w vs P',np.corrcoef(w,pc)[0,1],'w vs T',np.corrcoef(w,tc)[0,1])
print('P mass in top40 w coords',pc[order[:40]].sum()/pc.sum(),'T',tc[order[:40]].sum()/tc.sum())
```