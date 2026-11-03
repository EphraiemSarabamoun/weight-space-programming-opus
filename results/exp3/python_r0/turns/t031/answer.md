```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def sl(j): return slice(j*hd,(j+1)*hd)
def topV(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s
Vst=np.concatenate([W['W_v.0'][sl(j)] for j in [1,2,3,5]],0)
Th,sT=topV(Vst,27)
Pst=np.concatenate([W['W_q.0'][sl(j)] for j in [1,5]]+[W['W_k.0'][sl(j)] for j in [1,5]],0)
Ph,_=topV(Pst,30)
P=Ph-Th@(Th.T@Ph); P,_=np.linalg.qr(P)
b0=W['ln1_b.0']
Wq=W['W_q.0'][sl(0)]; Wk=W['W_k.0'][sl(0)]
M=Wq.T@Wk
# query side: all of space (since query content may be anything); key restricted to T
KT=M@Th   # 384 x 27 : query-dir x key-token
u,s,vt=np.linalg.svd(KT,full_matrices=False)
print("L0H0 query(any) x key-token sv:",s[:6].round(2))
xdir=vt[0]   # in T coords
qdir=u[:,0]
print("query dir of top: frac in T %.2f, in P %.2f"%(np.linalg.norm(Th.T@qdir)**2,np.linalg.norm(P.T@qdir)**2))
ub=Th.T@(Wk.T@(Wq@b0)); print("bias key pref |.|=%.2f cos with xdir %.2f"%(np.linalg.norm(ub),abs(ub@xdir)/np.linalg.norm(ub)))
# strength of xdir in various token readers vs. average over T directions
def gain(Mat,v): return np.linalg.norm(Mat@(Th@v))
rng=np.random.default_rng(0)
R=rng.standard_normal((27,500)); R/=np.linalg.norm(R,axis=0)
for name,Mat in [('Vstack',Vst)]+[(f'V{j}',W['W_v.0'][sl(j)]) for j in range(6)]+[(f'K{j}',W['W_k.0'][sl(j)]) for j in range(6)]+[(f'Q{j}',W['W_q.0'][sl(j)]) for j in range(6)]+[('fc0',W['W_fc.0'])]:
    gx=gain(Mat,xdir); gr=np.array([gain(Mat,R[:,i]) for i in range(500)])
    # rank of xdir gain among eigen directions of Mat restricted to T
    sv=np.linalg.svd(Mat@Th,compute_uv=False)
    print(f"{name}: gain(x)={gx:.2f}  random mean {gr.mean():.2f}  pct {np.mean(gr<gx)*100:.0f}  T-sv max {sv[0]:.2f} min {sv[-1]:.2f}")
```