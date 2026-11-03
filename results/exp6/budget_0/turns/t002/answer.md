```python
import numpy as np, json
def rc(p): return divmod(p,9)
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        a,b=r+dr,c+dc
        if 0<=a<9 and 0<=b<9: o.append(a*9+b)
    return o
NB=[nbrs(p) for p in range(81)]; DEG=[len(n) for n in NB]
KT={(0,1):1.0,(1,1):0.7,(0,2):0.5,(1,2):0.3,(2,2):0.15,(0,3):0.2,(1,3):0.1,(2,3):0.05,(0,4):0.05}
def K(p,q):
    (a,b),(c,d)=rc(p),rc(q); return KT.get(tuple(sorted((abs(a-c),abs(b-d)))),0.0)
def lf(*pairs):
    out={}
    for a,d in pairs:
        for k,v in d.items(): out[k]=out.get(k,0.0)+a*v
    return out
def C(v): return {'c':float(v)}
class Layer:
    def __init__(s,ind): s.ind=ind; s.units=[]
    def add(s,d):
        d=dict(d); b=d.pop('c',0.0); s.units.append((d,b)); return len(s.units)-1
    def addmany(s,f): return [s.add(f(p)) for p in range(81)]
    def build(s):
        assert len(s.units)<=512, len(s.units)
        W=np.zeros((s.ind,512),np.float32); b=np.zeros(512,np.float32)
        for j,(d,bb) in enumerate(s.units):
            for i,w in d.items(): W[i,j]+=w
            b[j]=bb
        return W,b
H=dict(tau=3.0,w=1.5,M=30.,wcap=25.,wef=60.,wesc=10.,wsa=10.,wpr=4.,winv=0.6,thinv=1.5,beta=0.3,wD=1.5,thD=1.0,cpass=0.5,BA=100.)
def one(arr,p): return {arr[p]:1.0}
def Ef(O,X,p): return lf((1,C(1)),(-1,one(O,p)),(-1,one(X,p)))
def nlib(O,X,s): return lf(*[(1,Ef(O,X,n)) for n in NB[s]])
def If(O,X,r):
    pr=[]
    for q in range(81):
        k=K(r,q)
        if k: pr+=[(k,one(O,q)),(-k,one(X,q))]
    return lf(*pr)
layers=[]
xO=[inp(p//9,p%9,0) for p in range(81)]; xX=[inp(p//9,p%9,1) for p in range(81)]
# L1
L=Layer(243); pO,pX=xO,xX
cO=L.addmany(lambda p: one(pO,p)); cX=L.addmany(lambda p: one(pX,p))
B0={col:L.addmany(lambda s,a=a: lf((1,nlib(pO,pX,s)),(4,one(a,s)),(1,C(-5)))) for col,a in (('O',pO),('X',pX))}
layers.append(L)
# L2
L=Layer(512); pO,pX=cO,cX
cO=L.addmany(lambda p: one(pO,p)); cX=L.addmany(lambda p: one(pX,p))
B1={col:L.addmany(lambda s,a=a,pv=B0[col]: lf((1,one(pv,s)),*[(1,one(pv,n)) for n in NB[s]],(16,one(a,s)),(1,C(-16)))) for col,a in (('O',pO),('X',pX))}
layers.append(L)
# L3
L=Layer(512); pO,pX=cO,cX
cO=L.addmany(lambda p: one(pO,p)); cX=L.addmany(lambda p: one(pX,p))
B2={col:L.addmany(lambda s,a=a,pv=B1[col]: lf((1,one(pv,s)),*[(1,one(pv,n)) for n in NB[s]],(64,one(a,s)),(1,C(-64)))) for col,a in (('O',pO),('X',pX))}
LB={col:L.addmany(lambda s,a=a: lf((1,nlib(pO,pX,s)),(4,one(a,s)),(1,C(-4)))) for col,a in (('O',pO),('X',pX))}
layers.append(L)
# L4
L=Layer(512); pO,pX=cO,cX
cO=L.addmany(lambda p: one(pO,p)); cX=L.addmany(lambda p: one(pX,p))
B3={col:L.addmany(lambda s,a=a,pv=B2[col],lb=LB[col]: lf((1,one(pv,s)),*[(1,one(pv,n)) for n in NB[s]],*[(1,one(lb,n)) for n in NB[s]],(400,one(a,s)),(1,C(-400)))) for col,a in (('O',pO),('X',pX))}
Pp=L.addmany(lambda r: If(pO,pX,r)); Nn=L.addmany(lambda r: lf((-1,If(pO,pX,r))))
layers.append(L)
# L5
L=Layer(512); pO,pX=cO,cX
cO=L.addmany(lambda p: one(pO,p)); cX=L.addmany(lambda p: one(pX,p))
AT={col:L.addmany(lambda q,a=a,b3=B3[col]: lf((1,one(a,q)),(-1,one(b3,q)))) for col,a in (('O',pO),('X',pX))}
G=L.addmany(lambda r: lf((1,C(H['w'])),(-1,one(Pp,r)),(-1,one(Nn,r)),(-H['M'],one(pO,r)),(-H['M'],one(pX,r))))
EF=L.addmany(lambda p: lf(*[(1,one(pO,n)) for n in NB[p]],(1,C(1-DEG[p]))))
layers.append(L)
# L6
L=Layer(512); pO,pX=cO,cX
def Af(p):
    pr=[(1,C(H['BA'])),(-H['wef'],one(EF,p)),(-H['beta'],If(pO,pX,p))]
    pr+=[(H['wcap'],one(AT['X'],q)) for q in NB[p]]
    for r in range(81):
        k=1.0 if r==p else K(p,r)
        if k: pr.append((k,one(G,r)))
    return lf(*pr)
A=L.addmany(Af)
def ep(p): return lf(*[(1,Ef(pO,pX,n)) for n in NB[p]])
def op(p): return lf(*[(1,one(pO,n)) for n in NB[p]])
def sAT(col,p): return lf(*[(1,one(AT[col],n)) for n in NB[p]])
Esc=L.addmany(lambda p: lf((2,sAT('O',p)),(1,ep(p)),(1,op(p)),(1,C(-4))))
SA=L.addmany(lambda p: lf((1,C(2)),(-1,ep(p)),(-1,op(p)),(1,sAT('O',p)),(-2,sAT('X',p))))
Pr=L.addmany(lambda q: lf((3,one(pX,q)),*[(1,one(pO,n)) for n in NB[q]],*[(-1,one(pX,n)) for n in NB[q]],(1,C(-DEG[q]))))
Inv=L.addmany(lambda p: lf((-1,If(pO,pX,p)),(1,C(-H['thinv']))))
D=L.addmany(lambda q: lf((1,If(pO,pX,q)),(H['M'],one(pX,q)),(1,C(-H['thD']-H['M']))))
layers.append(L)
for i,L in enumerate(layers): print('L%d units'%(i+1),len(L.units))
# output
t=H['tau']; oW=np.zeros((512,82),np.float32); ob=np.zeros(82,np.float32)
for p in range(81):
    r,c=rc(p); d=min(r,c,8-r,8-c); prior={0:-1.0,1:-0.3}.get(d,0.0)
    m=mv(r,c)
    oW[A[p],m]+=t; ob[m]=t*(-H['BA']+prior)
    oW[Esc[p],m]+=t*H['wesc']; oW[SA[p],m]+=-t*H['wsa']; oW[Inv[p],m]+=t*H['winv']
    for q in NB[p]:
        oW[Pr[q],m]+=t*H['wpr']; oW[D[q],m]+=t*H['wD']
ob[PASS]=t*H['cpass']
for i,L in enumerate(layers):
    W,b=L.build(); save('L%d.W'%(i+1),W); save('L%d.b'%(i+1),b)
    print(i+1,W.shape,int((W!=0).sum()),float(np.abs(W).max()),float(b.min()),float(b.max()))
save('out.W',oW); save('out.b',ob)
print(saved()); json.dump(H,open('/work/hparams_v1.json','w'))
```