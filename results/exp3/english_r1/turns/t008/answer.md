```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
outs=[];ins=[]
names=[0,1,4,5,2,3]
for h in names:
    Q,K,V,O=H[(0,h)]
    U,S,Vt=np.linalg.svd(O@V)
    outs.append(U[:,:3]); ins.append(Vt[:3])
    print(h,np.round(S[:5],2))
print('input-dir overlaps (top3 principal cosines)')
for a in range(6):
    print(names[a],[np.round(np.linalg.svd(ins[a]@ins[b].T,compute_uv=False),2).tolist() for b in range(6)])
print('output-dir overlaps')
for a in range(6):
    print(names[a],[np.round(np.linalg.svd(outs[a].T@outs[b],compute_uv=False),2).tolist() for b in range(6)])
```