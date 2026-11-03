```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
for h in range(6):
    Q,K,V,O=H[(0,h)]
    print(h,'V',np.round(np.linalg.svd(V,compute_uv=False)[:40],3))
    print(h,'OV',np.round(np.linalg.svd(O@V,compute_uv=False)[:40],3))
s=slice(128,192)
print('unfolded V 0.2',np.round(np.linalg.svd(W['W_v.0'][s],compute_uv=False)[:64],3))
```