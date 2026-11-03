```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
Q,K,V,O=H[(0,2)]; OV02=O@V
Uv,Sv,Vv=np.linalg.svd(OV02)
Q3,K3,_,_=H[(3,1)]
M=Q3.T@K3@OV02    # 384x384, query side x, key side x
ev=np.linalg.eigvals(K3@OV02@Q3.T)
ev=ev[np.argsort(-np.abs(ev))]
print('full composed eig (3.1 x 0.2):',' '.join('%.2f@%d'%(abs(z),np.degrees(np.angle(z))) for z in ev[:40]))
# symmetric part on token-ish subspace
for k in [20,30,40]:
    Tr=Vv[:k].T
    G=Tr.T@M@Tr; Gs=(G+G.T)/2
    e=np.linalg.eigvalsh(Gs)
    print(k,'sym eig',np.round(e[::-1],2))
```