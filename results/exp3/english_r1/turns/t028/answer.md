```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
for (l,h) in [(3,1),(3,5),(3,0),(3,2),(3,3),(3,4),(0,2)]:
    Q,K,V,O=H[(l,h)]
    s=np.linalg.svd(O@V,compute_uv=False); r=s[:-1]/s[1:]
    print(l,h,np.round(s[:40],2)); top=np.argsort(-r[5:50])[:3]+5; print('   knee idx',top,np.round(r[top],3))
for l in range(4):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False); r=s[:-1]/s[1:]; top=np.argsort(-r[5:80])[:3]+5
    print('proj',l,np.round(s[:40],1),top,np.round(r[top],3))
```