```python
import numpy as np
def Dl(l): return W[f'ln1_w.{l}']
def blk(nm,l,j):
    s=slice(64*j,64*j+64); return W[f'{nm}.{l}'][s]*Dl(l)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
U,S,Vt=np.linalg.svd(QK3)
P,_=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T]))
A=np.vstack([W['W_v.0'][0:64],W['W_v.0'][192:256]])
_,_,vt=np.linalg.svd(A); T=vt[:24].T
# output spaces of L0 heads (top singular left vectors of OV)
outsp={}
for j in range(6):
    s=slice(64*j,64*j+64)
    OV=W['W_o.0'][:,s]@blk('W_v',0,j)
    u,sv,_=np.linalg.svd(OV)
    k=max(1,int((sv>0.2*sv[0]).sum()))
    outsp[j]=u[:,:k]
    print('L0H',j,'OV out rank~',k, 'sv',sv[:6].round(2))
F=W['W_fc.0']*W['ln2_w.0']
Pr=W['W_proj.0']
for dim in [21,99,184,39,58]:
    w=Pr[dim]; idx=np.argsort(-np.abs(w))[:12]
    print('dim',dim,'top writers', [(int(n), round(float(w[n]),2)) for n in idx[:6]])
    for n in idx[:6]:
        f=F[n]; tot=np.linalg.norm(f)**2
        parts={'T':np.linalg.norm(T.T@f)**2/tot,'P':np.linalg.norm(P.T@f)**2/tot}
        for j in range(6): parts[f'o{j}']=np.linalg.norm(outsp[j].T@f)**2/tot
        sk=f[[21,39,58,99,184,383]]
        print('   n',n,'|fc| %.2f'%np.sqrt(tot), {k:round(float(v),2) for k,v in parts.items()}, 'sinkdims in', sk.round(2))
```