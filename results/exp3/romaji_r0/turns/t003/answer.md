```python
import numpy as np
d=384
def heads(l):
    out=[]
    wl = W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq = W[f'W_q.{l}'][s]*wl[None,:]
        Wk = W[f'W_k.{l}'][s]*wl[None,:]
        Wv = W[f'W_v.{l}'][s]*wl[None,:]
        Wo = W[f'W_o.{l}'][:,s]
        out.append(dict(QK=Wq.T@Wk, OV=Wo@Wv, Wq=Wq,Wk=Wk,Wv=Wv,Wo=Wo))
    return out
H = [heads(l) for l in range(4)]
def pos_score(M):
    ev = np.linalg.eigvals(M)
    return ev.real.sum()/np.abs(ev).sum()
print('OV positivity per head')
for l in range(4):
    print(l, [round(pos_score(H[l][j]['OV']),3) for j in range(6)])
print('QK positivity per head (direct same-direction matching)')
for l in range(4):
    print(l, [round(pos_score(H[l][j]['QK']),3) for j in range(6)])
# K-composition scores: ||QK^T? standard: K-comp = ||W_QK^T? use ||Wk Wo Wv|| / (||Wk|| ||OV||)
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l in range(1,4):
    print('layer',l,'K-comp with earlier heads (rows=this layer heads, cols=earlier layer heads)')
    for lp in range(l):
        Kc = np.array([[comp(H[l][j]['Wk'],H[lp][i]['OV']) for i in range(6)] for j in range(6)])
        Qc = np.array([[comp(H[l][j]['Wq'],H[lp][i]['OV']) for i in range(6)] for j in range(6)])
        Vc = np.array([[comp(H[l][j]['Wv'],H[lp][i]['OV']) for i in range(6)] for j in range(6)])
        print(' from layer',lp); print(' K\n',Kc.round(3)); print(' Q\n',Qc.round(3)); print(' V\n', Vc.round(3))
```