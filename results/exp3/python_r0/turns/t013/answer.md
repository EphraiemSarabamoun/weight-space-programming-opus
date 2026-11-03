```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def topU(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
def enr(R,U): return (np.linalg.norm(R@U)**2/U.shape[1])/(np.linalg.norm(R)**2/384)
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th=topV(Vst,27)
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph=topV(Pst,20)
Ex,_=np.linalg.qr(Th/g0[:,None]); Px,_=np.linalg.qr(Ph/g0[:,None])
Ex2,_=np.linalg.qr(Th*g0[:,None]); 
writers={'Ex':Ex,'Px':Px,'Thraw':Th,'A01':topU(OV(0,1),20),'A05':topU(OV(0,5),20),'A02':topU(OV(0,2),6),'A03':topU(OV(0,3),6),'A00':topU(OV(0,0),2),'A04':topU(OV(0,4),2)}
for l in range(1,3):
    for j in range(6): writers[f'A{l}{j}']=topU(OV(l,j),20)
print("overlap Ex vs A01:", np.linalg.norm(Ex.T@writers['A01'])**2/20, "Ex vs A05", np.linalg.norm(Ex.T@writers['A05'])**2/20, "Ex vs Px", np.linalg.norm(Ex.T@Px)**2/20)
wn=list(writers)
print("       ", " ".join(f"{w:>5}" for w in wn))
for l in range(1,4):
    for j in range(6):
        for kind in 'qkv':
            R=W[f'W_{kind}.{l}'][sl(j)]*W[f'ln1_w.{l}']
            vals=[enr(R,writers[w]) if (w[0]!='A' or int(w[1])<l) else np.nan for w in wn]
            print(f"L{l}H{j}{kind}", " ".join(f"{v:5.1f}" for v in vals))
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print(f"MLP{l} ", " ".join(f"{enr(F,writers[w]):5.1f}" if (w[0]!='A' or int(w[1])<=l) else '  nan' for w in wn))
```