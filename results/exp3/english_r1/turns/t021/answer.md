```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; M=Q.T@K
U,S,Vt=np.linalg.svd(M)
P=orth(np.hstack([U[:,:26],Vt[:26].T]),30)
Uv,Sv,Vv=np.linalg.svd(O@V); T=orth(Vv[:24].T-P@(P.T@Vv[:24].T),20)
kd={}
for h in [0,1,3,4,5]:
    Qh,Kh,Vh,Oh=H[(0,h)]
    Mh=Qh.T@Kh
    u,s,vt=np.linalg.svd(Mh)
    print(h,'QK sv',np.round(s[:4],3))
    for r in range(2):
        qd=u[:,r]; kdir=vt[r]
        print('   comp',r,'q in P %.2f T %.2f | k in P %.2f T %.2f'%(np.linalg.norm(P.T@qd),np.linalg.norm(T.T@qd),np.linalg.norm(P.T@kdir),np.linalg.norm(T.T@kdir)))
    kt=T@(T.T@vt[0]); kd[h]=kt/np.linalg.norm(kt)
    # key-token direction via K@T top
    _,_,vv=np.linalg.svd(Kh@T); kd[h]=T@vv[0]
names=list(kd)
print('key-token dir cosines:')
for a in names: print(a,[round(abs(float(kd[a]@kd[b])),2) for b in names])
# compare with 0.2 OV top input token dirs and fc0 top input
F=W['W_fc.0']*W['ln2_w.0']; f0=np.linalg.svd(F)[2][0]
print('fc0 top dir vs key-token dirs',[round(abs(float(f0@kd[a])),2) for a in names])
```