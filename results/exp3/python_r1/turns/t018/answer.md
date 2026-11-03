```python
import numpy as np
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        res.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
g0=W['ln1_w.0']; top=np.argsort(-g0)[:40]
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
b0=W['ln1_b.0']
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    M=Wq.T@Wk
    Uq,Sq,Vk=np.linalg.svd(M)
    print(f"L0H{j} sv {np.round(Sq[:4],2)}")
    for t in range(3):
        print(f"   q{t}: posfrac {(Uq[top,t]**2).sum():.2f} Sin {np.linalg.norm(Sin.T@Uq[:,t])**2:.2f} | k{t}: posfrac {(Vk[t,top]**2).sum():.2f} Sin {np.linalg.norm(Sin.T@Vk[t])**2:.2f}; PRq {1/np.sum(Uq[:,t]**4):.0f} PRk {1/np.sum(Vk[t]**4):.0f}")
    s=slice(64*j,64*j+64)
    qb=W['W_q.0'][s]@b0; kb=W['W_k.0'][s]@b0
    print(f"   |q bias| {np.linalg.norm(qb):.3f}  bias-query on keys: top-dir effect {np.linalg.norm(qb@Wk)/8:.3f}")
```