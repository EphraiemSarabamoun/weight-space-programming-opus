```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Q,K,V,O=H[(0,2)]; M=Q.T@K
U,S,Vt=np.linalg.svd(M)
k=20
X=orth(np.hstack([U[:,:k],Vt[:k].T]),k)
R=X.T@M@X
C=np.linalg.solve(R.T,R)
ev,EV=np.linalg.eig(C)
ang=np.degrees(np.angle(ev))/2
# dual basis: coordinates of a vector y (in X coords) in eigenbasis: EV^{-1} y
EVinv=np.linalg.inv(EV)
print('X captures of each head V-input / Q / K energy:')
for h in range(6):
    Qh,Kh,Vh,Oh=H[(0,h)]
    _,s,vt=np.linalg.svd(Oh@Vh); vin=vt[:3]  # top-3 input dirs
    cap=np.linalg.norm(vin@X,axis=1)
    coords=np.abs(EVinv@(X.T@vin.T))   # 20 x 3
    print(h,'V-in captured by X',np.round(cap,2))
    for a,c in sorted(zip(ang,coords[:,0]),key=lambda t:t[0]):
        pass
    order=np.argsort(ang)
    print('   angle:',' '.join('%d'%a for a in ang[order]))
    for d in range(3): print('   dir%d:'%d,' '.join('%.2f'%c for c in coords[order,d]/coords[:,d].max()))
```