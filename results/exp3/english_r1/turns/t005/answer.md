```python
for l in range(4):
    wl=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*wl; K=W[f'W_k.{l}'][s]*wl
        ev=np.linalg.eigvals(K@Q.T)
        idx=np.argsort(-np.abs(ev))[:10]
        e=ev[idx]
        M=Q.T@K; sym=(M+M.T)/2; asym=(M-M.T)/2
        trratio=np.real(ev.sum())/np.abs(ev).sum()
        print(l,h,'|sym|/|asym|=%.2f'%(np.linalg.norm(sym)/np.linalg.norm(asym)),'tr/sum|ev|=%.2f'%trratio,' '.join('%.1f@%d'%(abs(z),np.degrees(np.angle(z))) for z in e))
```