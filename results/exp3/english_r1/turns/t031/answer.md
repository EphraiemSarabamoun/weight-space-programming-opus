```python
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
pj=W['W_proj.3']; fc=W['W_fc.3']*W['ln2_w.3']
U,S,_=np.linalg.svd(pj)
print('proj3 sv',np.round(S[:36],2))
# overlap of 3.1 OV output with proj3 top-k
s=slice(64,128); OV=W['W_o.3'][:,s]@(W['W_v.3'][s]*W['ln1_w.3'])
Uo,So,_=np.linalg.svd(OV)
for k in [22,25,28,29,30]:
    print(k,'principal cos (3.1 OV top-k vs proj3 top-29):',np.round(np.linalg.svd(Uo[:,:k].T@U[:,:29],compute_uv=False)[[0,k//2,k-3,k-2,k-1]],2))
# energy of proj3 in each of its top dirs relative; also check proj2 spectrum after removing proj3 top29
for l in [2,1]:
    p=W[f'W_proj.{l}']; s2=np.linalg.svd(U[:,:29].T@p,compute_uv=False); print('proj',l,'in proj3 top29 sv',np.round(s2,1))
```