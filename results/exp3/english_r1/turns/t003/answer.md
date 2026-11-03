```python
w=W['ln1_w.0']
for l in range(4):
    wl=W[f'ln1_w.{l}']
    print('layer',l)
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*wl; K=W[f'W_k.{l}'][s]*wl; V=W[f'W_v.{l}'][s]*wl; O=W[f'W_o.{l}'][:,s]
        sq=np.linalg.svd(Q,compute_uv=False); sk=np.linalg.svd(K,compute_uv=False); sv=np.linalg.svd(V,compute_uv=False)
        OV=O@V; so=np.linalg.svd(OV,compute_uv=False)
        QK=Q.T@K; sqk=np.linalg.svd(QK,compute_uv=False)
        print(h,'Q',np.round(sq[:6],2),'K',np.round(sk[:6],2),'V',np.round(sv[:5],2),'OV',np.round(so[:6],2),'QK',np.round(sqk[:6],1))
```