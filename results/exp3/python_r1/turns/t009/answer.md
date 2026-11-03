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
def topin(M,k):  # top-k right singular vectors (input space), as columns
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T, S[:k]
def overlap(V,Wsub):  # fraction of V's energy in span of Wsub columns (orthonormal)
    Q,_=np.linalg.qr(Wsub); return np.linalg.norm(Q.T@V,axis=0)**2
_,_,Wv4,Wo4=H[0][4]; V4,S4=topin(Wo4@Wv4,2)
_,_,Wv0,Wo0=H[0][0]; V0,S0=topin(Wo0@Wv0,2)
print("cos between L0H4 and L0H0 OV inputs", np.round(V4.T@V0,3))
for j in [1,5,2,3]:
    Wq,Wk,Wv,Wo=H[0][j]
    QK=Wq.T@Wk
    U,S,Vt=np.linalg.svd(QK); P=np.hstack([U[:,:16],Vt[:16].T])
    Vv,_=topin(Wo@Wv,32)
    print(f"L0H{j}: L0H4-in in QKspace {np.round(overlap(V4,P),2)}  in Vspace {np.round(overlap(V4,Vv),2)} | L0H0-in QK {np.round(overlap(V0,P),2)} V {np.round(overlap(V0,Vv),2)}")
# baseline random 2 vectors overlap expectation with 32-dim subspace = 32/384
print(32/384)
# overlap of L0H1 QK-space with its V-space
for j in [1,5]:
    Wq,Wk,Wv,Wo=H[0][j]; U,S,Vt=np.linalg.svd(Wq.T@Wk); P=np.hstack([U[:,:16],Vt[:16].T])
    Vv,_=topin(Wo@Wv,16)
    print(j, np.round(overlap(Vv,P),2))
```