```python
import numpy as np
S=[21,39,58,99,184,383]
R=np.array([i for i in range(384) if i not in S])
def get(l,j,nm):
    s=slice(64*j,64*j+64)
    if nm=='Wo': return W[f'W_o.{l}'][:,s]
    return W[f'{nm}.{l}'][s]*W[f'ln1_w.{l}']
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
for l in range(1,4):
    for lp in range(l):
        print(f'== layer {l} heads (rows) reading layer {lp} heads (cols), S removed')
        for kind in ['W_k','W_q','W_v']:
            M=np.zeros((6,6)); B=np.zeros((6,6))
            for j in range(6):
                A=get(l,j,kind)[:,R]
                for i in range(6):
                    OV=(get(lp,i,'Wo')@get(lp,i,'W_v'))[R][:,R] if False else (get(lp,i,'Wo')[R]@get(lp,i,'W_v'))
                    M[j,i]=comp(A,OV)
                    # baseline: random orthogonal rotation of OV output space
                    Q,_=np.linalg.qr(rng.standard_normal((len(R),len(R))))
                    B[j,i]=comp(A,Q@OV)
            print(kind, '\n', M.round(3), '\n baseline mean %.3f sd %.3f'%(B.mean(),B.std()))
```