```python
outs=[];ins=[]
for h in [0,1,4,5,2,3]:
    Q,K,V,O=H[(0,h)]
    U,S,Vt=np.linalg.svd(O@V)
    outs.append(U[:,:3]); ins.append(Vt[:3])
    print(h,np.round(S[:5],2))
names=[0,1,4,5,2,3]
print('input-dir overlaps (top3 subspace principal cosines)')
for a in range(6):
    print(names[a],[np.round(np.linalg.svd(ins[a]@ins[b].T,compute_uv=False),2).tolist() for b in range(6)])
print('output-dir overlaps')
for a in range(6):
    print(names[a],[np.round(np.linalg.svd(outs[a].T@outs[b],compute_uv=False),2).tolist() for b in range(6)])
```