```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
OVs=np.hstack([H[(0,h)][3]@H[(0,h)][2] for h in [0,1,4,5]])
U,S,_=np.linalg.svd(OVs); print('broad out sv',np.round(S[:6],2)); B=U[:,:3]
U2,S2,_=np.linalg.svd(H[(0,2)][3]@H[(0,2)][2]); P2=U2[:,:20]
U3,S3,_=np.linalg.svd(H[(0,3)][3]@H[(0,3)][2]); P3=U3[:,:3]
print('overlap B vs 0.2 out',np.round(np.linalg.svd(B.T@P2,compute_uv=False),2))
for l in [1,2,3]:
  for h in range(6):
    Q,K,V,O=H[(l,h)]
    fq=np.linalg.norm(Q@B)**2/np.linalg.norm(Q)**2; fk=np.linalg.norm(K@B)**2/np.linalg.norm(K)**2
    fq2=np.linalg.norm(Q@P2)**2/np.linalg.norm(Q)**2; fk2=np.linalg.norm(K@P2)**2/np.linalg.norm(K)**2
    fv=np.linalg.norm(V@B)**2/np.linalg.norm(V)**2; fv2=np.linalg.norm(V@P2)**2/np.linalg.norm(V)**2
    M3=B.T@Q.T@K@B
    ev=np.linalg.eigvals(M3)
    print(l,h,'B: q %.2f k %.2f v %.2f | P2(0.2out): q %.2f k %.2f v %.2f'%(fq,fk,fv,fq2,fk2,fv2),'B^TQ^TKB ev',np.round(ev,1))
```