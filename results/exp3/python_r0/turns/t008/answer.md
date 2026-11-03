```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def eigplanes(l,j,k=10):
    A=W[f'W_q.{l}'][sl(j)]; B=W[f'W_k.{l}'][sl(j)]   # h-space bilinear
    ev,Y=np.linalg.eig(B@A.T)
    o=np.argsort(-np.abs(ev)); ev=ev[o]; Y=Y[:,o]
    X=A.T@Y
    res=[]; used=set()
    for i in range(len(ev)):
        if len(res)>=k: break
        if ev[i].imag<-1e-9: continue
        x=X[:,i]
        if abs(ev[i].imag)>1e-9:
            Pl=np.stack([x.real,x.imag],1)
        else:
            Pl=x.real[:,None]
        q,_=np.linalg.qr(Pl)
        res.append((ev[i],q))
    return res
E1=eigplanes(0,1,8); E5=eigplanes(0,5,10)
print("H1:",[f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e,_ in E1])
print("H5:",[f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e,_ in E5])
O=np.array([[np.linalg.norm(a.T@b)**2/min(a.shape[1],b.shape[1]) for _,b in E5] for _,a in E1])
print(O.round(2))
# also self-check: are eigvec planes of H1 near-invariant for H5's M?
```