```python
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
pj=W['W_proj.3']; Wo=W['W_o.3']
U,S,_=np.linalg.svd(pj); Up=U[:,:29]
print('W_o.3 sv',np.round(np.linalg.svd(Wo,compute_uv=False)[:40],2))
print('frac of W_o.3 energy in proj3 top-29 subspace',np.linalg.norm(Up.T@Wo)**2/np.linalg.norm(Wo)**2, 'baseline',29/384)
for h in range(6):
    s=slice(64*h,64*h+64); OV=Wo[:,s]@(W['W_v.3'][s]*W['ln1_w.3'])
    print(h,'OV energy in proj3 top29: %.2f'%(np.linalg.norm(Up.T@OV)**2/np.linalg.norm(OV)**2))
# combined layer3 output
comb=np.hstack([pj/np.linalg.norm(pj), Wo/np.linalg.norm(Wo)])
s=np.linalg.svd(comb,compute_uv=False); print('combined sv',np.round(s[:40]/s[0],3))
# also previous layers proj top-29 energy fraction
for l in range(3):
    P2=W[f'W_proj.{l}']; print('proj',l,'energy in proj3 top29: %.3f'%(np.linalg.norm(Up.T@P2)**2/np.linalg.norm(P2)**2))
print('ln weights at proj3 top dirs?')
```