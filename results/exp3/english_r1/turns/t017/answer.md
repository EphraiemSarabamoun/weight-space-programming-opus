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
F=W['W_fc.0']*W['ln2_w.0']
# token subspace estimate 2: MLP0 input minus P minus attention outputs? just print spectrum of F projected onto complement of P
Pc=np.eye(384)-P@P.T
print('F (fc0) on P-complement sv',np.round(np.linalg.svd(F@Pc,compute_uv=False)[:60],1))
print('F on T',np.round(np.linalg.svd(F@T,compute_uv=False),1))
# rank of unfolded layer0 attention inputs
A=np.vstack([W[n] for n in ['W_q.0','W_k.0','W_v.0']])
print('unfolded qkv0 sv',np.round(np.linalg.svd(A,compute_uv=False)[:90],2))
```