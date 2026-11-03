```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; M=Q.T@K
Vin=orth(np.hstack([np.linalg.svd(H[(0,h)][3]@H[(0,h)][2])[2][:2].T for h in [0,1,4,5]]),4)
print('sv of stacked broad V-in dirs', np.round(np.linalg.svd(np.hstack([np.linalg.svd(H[(0,h)][3]@H[(0,h)][2])[2][:2].T for h in [0,1,4,5]]),compute_uv=False),2))
Bv=Vin[:,:2]
R2=Bv.T@M@Bv
print('2x2 restricted 0.2 form on broad V-in plane:\n',np.round(R2,2), 'eig',np.linalg.eigvals(R2), 'angle',np.degrees(np.angle(np.linalg.eigvals(R2))))
# same for 0.3 and broad heads QK
for h in [3,0,1,4,5]:
    Qh,Kh,_,_=H[(0,h)]
    print(h,np.round(Bv.T@Qh.T@Kh@Bv,3))
# linear eigenvectors of M (top), overlap with Bv
ev,EV=np.linalg.eig(K@Q.T)   # 64x64: eigvecs in head space; map to residual: Q.T@EV gives left?, use M eigen directly
evM,EVM=np.linalg.eig(M)
idx=np.argsort(-np.abs(evM))[:20]
for i in idx:
    v=EVM[:,i]; plane=orth(np.stack([v.real,v.imag],1),2)
    print('ev %.2f@%d'%(abs(evM[i]),np.degrees(np.angle(evM[i]))),'overlap with Bv',np.round(np.linalg.svd(Bv.T@plane,compute_uv=False),2))
```