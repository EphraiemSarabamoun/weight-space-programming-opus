```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; OV02=O@V
U,S,Vt=np.linalg.svd(Q.T@K)
P=orth(np.hstack([U[:,:26],Vt[:26].T]),30)
Uv,Sv,Vv=np.linalg.svd(OV02); T=orth(Vv[:24].T-P@(P.T@Vv[:24].T),20)
for h in range(6):
    Q,K,V,O=H[(0,h)]
    M=Q.T@K
    for nm,A,Bm in [('PP',P,P),('PT',P,T),('TP',T,P),('TT',T,T)]:
        s=np.linalg.svd(A.T@M@Bm,compute_uv=False)
        print(h,nm,np.round(s[:5],2))
    # key reading in T: singular values
    print(h,'K@T sv',np.round(np.linalg.svd(K@T,compute_uv=False)[:5],2),'Q@T',np.round(np.linalg.svd(Q@T,compute_uv=False)[:5],2),'K@P',np.round(np.linalg.svd(K@P,compute_uv=False)[:5],2),'Q@P',np.round(np.linalg.svd(Q@P,compute_uv=False)[:5],2))
# fc0
F=W['W_fc.0']*W['ln2_w.0']
print('fc0 frac P %.3f T %.3f'%(np.linalg.norm(F@P)**2/np.linalg.norm(F)**2, np.linalg.norm(F@T)**2/np.linalg.norm(F)**2))
u=np.linalg.svd(F)[2][0]; print('fc0 top dir in P %.2f T %.2f'%(np.linalg.norm(P.T@u),np.linalg.norm(T.T@u)))
```