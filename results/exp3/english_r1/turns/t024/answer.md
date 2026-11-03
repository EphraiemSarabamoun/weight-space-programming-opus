```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
fc=W['W_fc.0']*W['ln2_w.0']; pj=W['W_proj.0']
act=np.linalg.norm(pj,axis=0)>0.5
print('active',act.sum())
A=fc[act]
print('active fc sv',np.round(np.linalg.svd(A,compute_uv=False)[:80],2))
dead=fc[~act]; print('dead fc sv',np.round(np.linalg.svd(dead,compute_uv=False)[:10],2))
# remove attention-output subspaces of layer0
outs=np.hstack([orth(H[(0,h)][3]@H[(0,h)][2],k) for h,k in [(0,3),(1,3),(4,3),(5,3),(2,30),(3,6)]])
Ao=orth(outs,None if False else outs.shape[1])
Ao=orth(outs,np.linalg.matrix_rank(outs,tol=1e-3))
print('attn out subspace dim',Ao.shape[1])
Pr=np.eye(384)-Ao@Ao.T
print('active fc w/o attn-out sv',np.round(np.linalg.svd(A@Pr,compute_uv=False)[:60],2))
```