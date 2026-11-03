```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def Mqk(l,j):
    g=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][sl(j)]*g).T@(W[f'W_k.{l}'][sl(j)]*g)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
def topU(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
g0=W['ln1_w.0']
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th=topV(Vst,27)
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph=topV(Pst,24)
E,_=np.linalg.qr(Th/(g0[:,None]+0.05))
P,_=np.linalg.qr(Ph/(g0[:,None]+0.05))
# make P orthogonal to E
P=P-E@(E.T@P); P,_=np.linalg.qr(P)
S=topU(OV(0,4),2)
A5=topU(OV(0,5),20); A1=topU(OV(0,1),20)
blocks={'E':E,'P':P,'S':S,'A5':A5,'A1':A1}
print("principal cos E-P",np.linalg.svd(E.T@P,compute_uv=False)[:3], "E-S",np.linalg.svd(E.T@S,compute_uv=False)[:2],"P-S",np.linalg.svd(P.T@S,compute_uv=False)[:2])
names=list(blocks)
for l in range(1,4):
    for j in range(6):
        M=Mqk(l,j); tot=np.linalg.norm(M)
        out=[]
        for qn in names:
            for kn in names:
                Bq=blocks[qn]; Bk=blocks[kn]
                v=np.linalg.norm(Bq.T@M@Bk)/np.sqrt(Bq.shape[1]*Bk.shape[1])
                out.append(v)
        out=np.array(out).reshape(len(names),len(names))
        print(f"L{l}H{j} tot/384={tot/384:.4f}  rows q: {names}, cols k")
        print((out/(tot/384)).round(1))
```