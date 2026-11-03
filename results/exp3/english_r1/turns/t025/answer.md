```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
fc=W['W_fc.0']*W['ln2_w.0']; pj=W['W_proj.0']
act=np.linalg.norm(pj,axis=0)>0.5; A=fc[act]
Q,K,V,O=H[(0,2)]; M=Q.T@K
U,S,Vt=np.linalg.svd(M)
for kp in [0,10,20,30]:
  for (k2,kb) in [(20,3),(30,3),(40,3)]:
    parts=[orth(H[(0,h)][3]@H[(0,h)][2],kb) for h in [0,1,4,5]]+[orth(O@V,k2),orth(H[(0,3)][3]@H[(0,3)][2],4)]
    if kp>0: parts.append(orth(np.hstack([U[:,:kp],Vt[:kp].T]),kp))
    S_=np.hstack(parts); Ao=orth(S_,np.linalg.matrix_rank(S_,tol=1e-3))
    Pr=np.eye(384)-Ao@Ao.T
    s=np.linalg.svd(A@Pr,compute_uv=False)
    r=s[:-1]/s[1:]
    print('P dims',kp,'0.2out dims',k2,'removed',Ao.shape[1],'sv',np.round(s[18:36],2),'largest ratio idx (10..45):',10+np.argmax(r[10:45]),np.round(r[10:45].max(),3))
```