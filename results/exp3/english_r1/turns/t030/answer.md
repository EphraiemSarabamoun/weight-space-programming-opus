```python
def head(l,h):
    s=slice(64*h,64*h+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
H={(l,h):head(l,h) for l in range(4) for h in range(6)}
def orth(M,k):
    U,S,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
OVs={k:H[k][3]@H[k][2] for k in H}
Bmap=sum(OVs[(0,h)] for h in [0,1,4,5])
def evinfo(M):
    ev=np.linalg.eigvals(M); i=np.argsort(-np.abs(ev))[:4]
    return ' '.join('%.2f@%d'%(abs(z),np.degrees(np.angle(z))) for z in ev[i])
for l in [1,2,3]:
    for h in range(6):
        Q,K,V,O=H[(l,h)]; M=Q.T@K
        BB=Bmap.T@M@Bmap   # positional phase (B) on both sides, in layer-0 input coords
        L1=sum(OVs[(1,hh)] for hh in range(6)); 
        s_bb=np.linalg.norm(BB)
        # matching on layer-1 outputs and layer-2 outputs (both sides)
        out=[]
        for ll in range(1,l):
            for hh in range(6):
                A=OVs[(ll,hh)]; G=A.T@M@A; ev=np.linalg.eigvals(G); out.append('%d.%d:%+.2f'%(ll,hh,np.real(ev).sum()/np.abs(ev).sum()))
        print(f'{l}.{h} |BB|=%.1f BBeig: %s'%(s_bb,evinfo(BB)), ' | self-match tr on later-head outputs:',' '.join(out))
```