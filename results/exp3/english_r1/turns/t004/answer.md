```python
w=W['ln1_w.0']
order=np.argsort(-w)
print(np.round(w[order[:60]],2))
for h in range(6):
    s=slice(64*h,64*h+64)
    for nm in ['W_q.0','W_k.0','W_v.0']:
        M=W[nm][s]*w
        cn=np.linalg.norm(M,axis=0)
        top=np.argsort(-cn)[:12]
        frac=np.sort(cn**2)[::-1].cumsum()/ (cn**2).sum()
        print(h,nm,'top dims',top,'cn',np.round(cn[top],2),'n for 50/90%',np.searchsorted(frac,0.5)+1,np.searchsorted(frac,0.9)+1)
```