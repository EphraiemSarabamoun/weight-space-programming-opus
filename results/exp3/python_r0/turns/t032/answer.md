```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,sT=topV(Vst,27)
rng=np.random.default_rng(0)
R=rng.standard_normal((27,1000)); R/=np.linalg.norm(R,axis=0)
def pct(Mat,v):
    gx=np.linalg.norm(Mat@(Th@v)); gr=np.linalg.norm(Mat@(Th@R),axis=0); return np.mean(gr<gx)*100, gx/gr.mean()
readers={'Vst':Vst,'V5':W['W_v.0'][sl(5)],'V1':W['W_v.0'][sl(1)],'fc0':W['W_fc.0'],'V4':W['W_v.0'][sl(4)]}
dirs={}
for j in [0,2,3]:
    M=W['W_q.0'][sl(j)].T@W['W_k.0'][sl(j)]
    u,s,vt=np.linalg.svd(M@Th,full_matrices=False)
    for r in range(3):
        dirs[f"H{j}k{r}"]=vt[r]
        print(f"L0H{j} key-token dir {r} sv {s[r]:.2f}:", "  ".join(f"{n} {pct(Mt,vt[r])[0]:.0f}%({pct(Mt,vt[r])[1]:.2f})" for n,Mt in readers.items()))
# Vstack principal token directions themselves: top and bottom
names=list(dirs)
C=np.array([[abs(dirs[a]@dirs[b]) for b in names] for a in names]); print(names); print(C.round(2))
```