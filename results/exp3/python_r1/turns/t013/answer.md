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
_,_,Wv4,Wo4=H[0][4]
U,S,Vt=np.linalg.svd(Wo4@Wv4); Sin=Vt[:2].T; Sout=U[:,:2]
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        fk=np.linalg.norm(Wk@Sout)**2/np.linalg.norm(Wk)**2
        fq=np.linalg.norm(Wq@Sout)**2/np.linalg.norm(Wq)**2
        fv=np.linalg.norm(Wv@Sout)**2/np.linalg.norm(Wv)**2
        # QK restricted: query from all, key from Sout: sv of Wq^T Wk Sout
        M=Wq.T@(Wk@Sout)
        sv=np.linalg.svd(M,compute_uv=False)
        Mall=np.linalg.svd(Wq.T@Wk,compute_uv=False)
        print(f"L{l}H{j} frac k in Sout {fk:.3f} q {fq:.3f} v {fv:.3f}  sv(QK->Sout) {np.round(sv,2)} topsv QK {np.round(Mall[:4],2)}")
g2=W['ln2_w.0']; F=W['W_fc.0']*g2
print("fc0 frac in Sin", np.linalg.norm(F@Sin)**2/np.linalg.norm(F)**2, "Sout", np.linalg.norm(F@Sout)**2/np.linalg.norm(F)**2)
for l in range(1,4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print(l,"fc frac in Sin", np.linalg.norm(F@Sin)**2/np.linalg.norm(F)**2, "Sout", np.linalg.norm(F@Sout)**2/np.linalg.norm(F)**2)
```