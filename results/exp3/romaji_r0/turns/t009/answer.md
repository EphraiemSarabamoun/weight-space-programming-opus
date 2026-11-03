```python
import numpy as np
def QKmat(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][s]*wl).T@(W[f'W_k.{l}'][s]*wl)
QK3=QKmat(0,3)
U,S,Vt=np.linalg.svd(QK3)
k=16
P,_=np.linalg.qr(np.hstack([U[:,:k],Vt[:k].T]))
r=P.shape[1]
print('P dim',r)
# remove sink dims from P? check P's weight on S dims
Sd=[21,39,58,99,184,383]
print('P mass on sink dims', (P[Sd]**2).sum().round(3))
rng=np.random.default_rng(1)
for l in range(4):
    row=[]
    for j in range(6):
        M=QKmat(l,j)
        f=np.linalg.norm(P.T@M@P)**2/np.linalg.norm(M)**2
        fq=np.linalg.norm(P.T@M)**2/np.linalg.norm(M)**2
        fk=np.linalg.norm(M@P)**2/np.linalg.norm(M)**2
        row.append(f'{f:.3f}/{fq:.2f}/{fk:.2f}')
    print('L',l,row)
print('random expectation PP %.4f, one side %.3f'%((r/384)**2, r/384))
```