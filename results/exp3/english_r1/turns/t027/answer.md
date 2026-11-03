```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
fc=W['W_fc.0']*W['ln2_w.0']; pj=W['W_proj.0']
Q,K,V,O=H[(0,2)]
parts=[orth(H[(0,h)][3]@H[(0,h)][2],3) for h in [0,1,4,5]]+[orth(O@V,30),orth(H[(0,3)][3]@H[(0,3)][2],4)]
S_=np.hstack(parts); Ao=orth(S_,np.linalg.matrix_rank(S_,tol=1e-3)); Pr=np.eye(384)-Ao@Ao.T
cn=np.linalg.norm(pj,axis=0)
for thr in [0.4,0.8,1.2,1.6,2.0]:
    act=cn>thr; A=fc[act]
    s=np.linalg.svd(A@Pr,compute_uv=False); r=s[:-1]/s[1:]
    top=np.argsort(-r[5:60])[:3]+5
    print('thr',thr,'n',act.sum(),'top ratio idx',top,np.round(r[top],3))
# weight by proj norm (effective contribution)
A=fc*cn[:,None]
s=np.linalg.svd(A@Pr,compute_uv=False); r=s[:-1]/s[1:]
print('weighted sv',np.round(s[:40],1)); top=np.argsort(-r[5:60])[:3]+5; print(top,np.round(r[top],3))
# without removing anything
s=np.linalg.svd(fc[cn>0.5],compute_uv=False); r=s[:-1]/s[1:]; top=np.argsort(-r[5:60])[:3]+5; print('raw',top,np.round(r[top],3))
```