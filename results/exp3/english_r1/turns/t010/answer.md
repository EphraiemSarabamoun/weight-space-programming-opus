```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
Q,K,V,O=H[(0,2)]
M=Q.T@K
U,S,Vt=np.linalg.svd(M)
print('QK sv',np.round(S[:40],2))
for k in [5,10,20,30]:
    print(k,'principal cos between Q-side and K-side top-k', np.round(np.linalg.svd(U[:,:k].T@Vt[:k].T,compute_uv=False),2))
# overlap with broad-head V input subspace
_,_,Vb=np.linalg.svd(H[(0,0)][3]@H[(0,0)][2]); Vb=Vb[:3]
print('broad V-input in Q-side top20', np.round(np.linalg.norm(Vb@U[:,:20],axis=1),2),'K-side',np.round(np.linalg.norm(Vb@Vt[:20].T,axis=1),2))
# 0.2 OV: input/out overlap
OV=O@V; Uo,So,Vo=np.linalg.svd(OV)
print('0.2 OV in/out overlap',np.round(np.linalg.svd(Uo[:,:20].T@Vo[:20].T,compute_uv=False),2))
print('0.2 OV input vs QK sides', np.round(np.linalg.svd(Vo[:20]@U[:,:20],compute_uv=False),2), np.round(np.linalg.svd(Vo[:20]@Vt[:20].T,compute_uv=False),2))
```