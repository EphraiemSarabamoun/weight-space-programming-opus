```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*wl; K=W[f'W_k.{l}'][s]*wl; V=W[f'W_v.{l}'][s]*wl; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def evstats(M):
    ev=np.linalg.eigvals(M); a=np.abs(ev)
    return np.real(ev).sum()/a.sum(), a.max()
for l2 in range(1,4):
    for h2 in range(6):
        Q,K,_,_=H[(l2,h2)]
        row=[]
        for l1 in range(l2):
            for h1 in range(6):
                _,_,V1,O1=H[(l1,h1)]
                OV=O1@V1
                # K-composition matching: query reads x directly, key reads OV x
                M=Q.T@(K@OV)
                # restrict to 64x64: eigenvalues of (K OV Q^T)
                ev=np.linalg.eigvals(K@OV@Q.T); a=np.abs(ev)
                kc=np.linalg.norm(K@OV)/np.linalg.norm(K)/np.linalg.norm(OV)
                row.append('%d.%d:%.2f/%+.2f'%(l1,h1,kc*10,np.real(ev).sum()/a.sum()))
        print(f'{l2}.{h2} K-comp*10/evtrace', ' '.join(row))
```