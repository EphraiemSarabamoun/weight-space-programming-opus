```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def OV(l,j): return W[f'W_o.{l}'][:,sl(j)]@(W[f'W_v.{l}'][sl(j)]*W[f'ln1_w.{l}'])
def topU(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def enr(R,U): return (np.linalg.norm(R@U)**2/U.shape[1])/(np.linalg.norm(R)**2/384)
writers={}
for l in range(3):
    for j in range(6):
        writers[f"A{l}{j}"]=topU(OV(l,j),4)
    writers[f"M{l}"]=topU(W[f'W_proj.{l}'],16)
wn=list(writers)
print("     ", " ".join(f"{w:>5}" for w in wn))
for l in range(1,4):
    for j in range(6):
        for kind in 'qkv':
            R=W[f'W_{kind}.{l}'][sl(j)]*W[f'ln1_w.{l}']
            vals=[enr(R,writers[w]) if int(w[1])<l else np.nan for w in wn]
            print(f"L{l}H{j}{kind}", " ".join(f"{v:5.1f}" for v in vals))
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    vals=[enr(F,writers[w]) if int(w[1])<=l and not (w[0]=='M' and int(w[1])==l) else np.nan for w in wn]
    print(f"MLP{l} ", " ".join(f"{v:5.1f}" for v in vals))
F=W['W_fc.0']*W['ln2_w.0']
print("MLP0 ", " ".join(f"{enr(F,writers[w]):5.1f}" if w[1]=='0' and w[0]=='A' else '  nan' for w in wn))
```