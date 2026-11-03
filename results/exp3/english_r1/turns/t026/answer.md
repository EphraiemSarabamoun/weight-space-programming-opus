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
# source maps (residual-writing linear maps, normalized to unit Frobenius-ish scale per output dir)
src={'T':T,'P':P}
for l in range(3):
    for h in range(6):
        Qh,Kh,Vh,Oh=H[(l,h)]; src[f'{l}.{h}']=Oh@Vh
    src[f'mlp{l}']=W[f'W_proj.{l}']
def blocknorm(Ml, A, B):
    return np.linalg.norm(A.T@Ml@B)
for (l,h) in [(3,0),(3,1),(3,5),(2,0),(1,0)]:
    Qh,Kh,_,_=H[(l,h)]; Mq=Qh.T@Kh
    names=[n for n in src if (n in 'TP') or (n.startswith('mlp') and int(n[3])<l) or ('.' in n and int(n[0])<l)]
    res=[]
    for a in names:
        for b in names:
            Aa=src[a]/np.linalg.norm(src[a]); Bb=src[b]/np.linalg.norm(src[b])
            res.append((blocknorm(Mq,Aa,Bb),a,b))
    res.sort(reverse=True)
    print(f'{l}.{h} top QK blocks (q-src,k-src):',' '.join('%s>%s:%.3f'%(a,b,v) for v,a,b in res[:14]))
```