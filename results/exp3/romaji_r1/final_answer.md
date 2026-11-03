
<invoke name="python">
<parameter name="code">
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(np.abs(a).mean()), float(a.std()), float(a.mean()))
</parameter>
</invoke>

<invoke name="python">
<parameter name="code">
d=384;hd=64
def sv(M,k=8): return np.linalg.svd(M,compute_uv=False)[:k]
for l in range(4):
    print("Layer",l)
    D=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Wq=W[f'W_q.{l}'][s]*D; Wk=W[f'W_k.{l}'][s]*D; Wv=W[f'W_v.{l}'][s]*D; Wo=W[f'W_o.{l}'][:,s]
        B=Wq.T@Wk
        Bs=(B+B.T)/2; Ba=(B-B.T)/2
        OV=Wo@Wv
        ev=np.linalg.eigvals(OV)
        print(h, "q",np.round(sv(Wq,4),2),"k",np.round(sv(Wk,4),2),"QK",np.round(sv(B,4),2),
              "sym/anti %.2f %.2f"%(np.linalg.norm(Bs),np.linalg.norm(Ba)),
              "OV",np.round(sv(OV,3),2), "OVtr/|| %.2f"%(np.trace(OV)/np.linalg.norm(OV)))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for l in range(4):
    D=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Wq=W[f'W_q.{l}'][s]*D; Wk=W[f'W_k.{l}'][s]*D
        M=Wk@Wq.T  # 64x64, same nonzero eigs as Wq.T@Wk
        ev=np.linalg.eigvals(M)
        idx=np.argsort(-np.abs(ev))[:12]
        print(l,h," ".join("%.1f@%d"%(abs(e),round(np.degrees(np.angle(e)))) for e in ev[idx]))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
def topright(M,r):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:r].T, S[:r]
def overlap(A,B):
    s=np.linalg.svd(A.T@B,compute_uv=False); return s
l=0;D=W['ln1_w.0']
Vall=np.vstack([W['W_v.0'][64*h:64*h+64]*D for h in range(6)])
Vsp,_=topright(Vall,28)
FCsp,_=topright(W['W_fc.0']*W['ln2_w.0'],40)
print("Wv0 svals",np.round(np.linalg.svd(Vall,compute_uv=False)[:40],2))
print("Wfc0 svals",np.round(np.linalg.svd(W['W_fc.0']*W['ln2_w.0'],compute_uv=False)[:40],2))
for h in range(6):
    s=slice(64*h,64*h+64)
    Q,_=topright(W['W_q.0'][s]*D,12); K,_=topright(W['W_k.0'][s]*D,12); V,_=topright(W['W_v.0'][s]*D,12)
    print(h,"Q-K",np.round(overlap(Q,K)[:6],2),"Q-Vall",np.round(overlap(Q,Vsp)[:6],2),"K-Vall",np.round(overlap(K,Vsp)[:6],2))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
D=W['ln1_w.0']
planes={}
for h in range(6):
    s=slice(64*h,64*h+64)
    Wq=W['W_q.0'][s]*D; Wk=W['W_k.0'][s]*D
    B=Wq.T@Wk
    # eigen of B via 64x64: B = Wq^T Wk ; eig of Wk Wq^T: M v = lam v -> B (Wq^T v) = lam Wq^T v
    M=Wk@Wq.T
    ev,V=np.linalg.eig(M)
    idx=np.argsort(-np.abs(ev))[:16]
    pl=[]
    for i in idx:
        if ev[i].imag<-1e-6: continue
        u=Wq.T@V[:,i]
        P=np.stack([u.real,u.imag],1)
        Qm,_=np.linalg.qr(P)
        pl.append((ev[i],Qm))
    planes[h]=pl
for h in range(6):
    for h2 in range(h+1,6):
        for (e1,P1) in planes[h][:5]:
            for (e2,P2) in planes[h2][:5]:
                o=np.linalg.svd(P1.T@P2,compute_uv=False)
                if o.min()>0.5:
                    print(h,h2,"%.1f@%d"%(abs(e1),np.degrees(np.angle(e1))),"%.1f@%d"%(abs(e2),np.degrees(np.angle(e2))),np.round(o,2))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
subs={}
for h in range(6):
    s=slice(64*h,64*h+64)
    Wq=W['W_q.0'][s]*D; Wk=W['W_k.0'][s]*D
    subs[h]=(topright(Wq,8)[0],topright(Wk,8)[0])
for h in range(6):
    print(h," ".join("%d:Q%.2f/K%.2f"%(h2,np.linalg.svd(subs[h][0].T@subs[h2][0],compute_uv=False)[:4].mean(),np.linalg.svd(subs[h][1].T@subs[h2][1],compute_uv=False)[:4].mean()) for h2 in range(6)))
# random baseline: 8-dim subspaces in 384
R1=np.linalg.qr(np.random.randn(384,8))[0];R2=np.linalg.qr(np.random.randn(384,8))[0]
print(np.linalg.svd(R1.T@R2,compute_uv=False)[:4].mean())
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
def QK(l,h):
    D=W[f'ln1_w.{l}'];s=slice(64*h,64*h+64)
    return (W[f'W_q.{l}'][s]*D).T@(W[f'W_k.{l}'][s]*D)
def OV(l,h):
    D=W[f'ln1_w.{l}'];s=slice(64*h,64*h+64)
    return W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*D)
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
QKs={(l,h):QK(l,h) for l in range(4) for h in range(6)}
OVs={(l,h):OV(l,h) for l in range(4) for h in range(6)}
for l2 in range(1,4):
    print("=== later layer",l2)
    for h2 in range(6):
        rowsK=[];rowsQ=[];rowsV=[]
        for l1 in range(l2):
            for h1 in range(6):
                rowsK.append(comp(QKs[(l2,h2)],OVs[(l1,h1)]))
                rowsQ.append(comp(QKs[(l2,h2)].T,OVs[(l1,h1)]))
                rowsV.append(comp(OVs[(l2,h2)],OVs[(l1,h1)]))
        print(h2,"K",np.round(rowsK,2))
        print("  Q",np.round(rowsQ,2))
        print("  V",np.round(rowsV,2))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
def eigsum(M,k=10):
    ev=np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(ev))[:k]
    return " ".join("%.2f@%d"%(abs(e),round(np.degrees(np.angle(e)))) for e in ev[idx]), np.sum(ev.real)/np.sum(np.abs(ev))
for (l2,h2) in [(1,4),(1,3),(1,1),(2,2),(2,4),(1,0),(2,0),(3,0),(3,1)]:
    for h1 in [1,3,5,0,2,4]:
        M=QKs[(l2,h2)]@OVs[(0,h1)]
        s,r=eigsum(M,8)
        print((l2,h2),"<- L0H%d"%h1,"comp %.2f"%comp(QKs[(l2,h2)],OVs[(0,h1)]),"realfrac %.2f |"%r,s)
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
def cosm(A,B): return np.sum(A*B)/np.linalg.norm(A)/np.linalg.norm(B)
B0=[QKs[(0,h)] for h in range(6)]
print(np.round([[cosm(B0[a],B0[b]) for b in range(6)] for a in range(6)],2))
print("vs squares")
print(np.round([[cosm(B0[a],B0[b]@B0[b]) for b in range(6)] for a in range(6)],2))
print("vs products")
for a in range(6):
    print(a,[(b,c,round(cosm(B0[a],B0[b]@B0[c]),2)) for b in range(6) for c in range(6) if abs(cosm(B0[a],B0[b]@B0[c]))>0.1])
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
print("ln1_w.0 stats",W['ln1_w.0'].mean(),W['ln1_w.0'].std(), "ln1_b.0 norm",np.linalg.norm(W['ln1_b.0']))
OVp=OVs[(0,1)]+OVs[(0,3)]+OVs[(0,5)]
for (l2,h2) in [(1,4),(1,3),(1,1),(2,2),(2,4)]:
    M=QKs[(l2,h2)]@OVp
    ev=np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(ev))
    ev=ev[idx]
    print((l2,h2)," ".join("%.2f%s"%(abs(e),"" if abs(e.imag)<1e-3*abs(e)+1e-9 and e.real>0 else ("-" if abs(e.imag)<1e-9 else "c")) for e in ev[:45]))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for (l2,h2) in [(1,4),(1,3),(1,1),(2,2),(2,4)]:
    for name,O in [("sum",OVp),("H1",OVs[(0,1)]),("H5",OVs[(0,5)]),("H3",OVs[(0,3)])]:
        M=QKs[(l2,h2)]@O
        S=(M+M.T)/2
        e=np.linalg.eigvalsh(S)
        pos=np.sort(e[e>0])[::-1]; neg=np.sort(-e[e<0])[::-1]
        print((l2,h2),name,"pos",np.round(pos[:32],2))
        print("        neg",np.round(neg[:12],2))
    print()
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        s=np.linalg.svd(B,compute_uv=False)
        Bs=(B+B.T)/2;Ba=(B-B.T)/2
        es=np.linalg.eigvalsh(Bs)
        O=OVs[(l,h)]; so=np.linalg.svd(O,compute_uv=False)
        evo=np.linalg.eigvals(O)
        print(l,h,"QK sv",np.round(s[:5],1),"F %.1f sym %.1f anti %.1f symeig+ %.1f symeig- %.1f"%(np.linalg.norm(B),np.linalg.norm(Bs),np.linalg.norm(Ba),es.max(),-es.min()),
              "| OV sv",np.round(so[:3],2),"F %.2f tr/sum|ev| %.2f"%(np.linalg.norm(O),evo.real.sum()/np.abs(evo).sum()))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
import io
rows=[]
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        s=np.linalg.svd(B,compute_uv=False)
        Bs=(B+B.T)/2;Ba=(B-B.T)/2
        es=np.linalg.eigvalsh(Bs)
        ev=np.linalg.eigvals(Wk:=None) if False else None
        print("L%dH%d  QKsv1 %.1f sv5 %.1f  sym %.1f anti %.1f  maxEigSym %.1f minEigSym %.1f"%(l,h,s[0],s[4],np.linalg.norm(Bs),np.linalg.norm(Ba),es.max(),es.min()))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
def angstats(B):
    # eigenvalues of B (rank<=64) via 64x64
    ev=np.linalg.eigvals(B)
    idx=np.argsort(-np.abs(ev))[:64]; ev=ev[idx]
    w=np.abs(ev)
    return (w*np.abs(np.angle(ev))).sum()/w.sum()*180/np.pi, (w*(ev.real>0)*(np.abs(ev.imag)<1e-6*w)).sum()/w.sum()
prevO=OVs[(0,1)]+OVs[(0,3)]+OVs[(0,5)]
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        ma,pr=angstats(B)
        s=np.linalg.svd(B,compute_uv=False)
        line="L%dH%d |QK|=%.1f sv1=%.1f meanAngle=%.0f posRealFrac=%.2f"%(l,h,np.linalg.norm(B),s[0],ma,pr)
        if l>0:
            M=B@prevO; e=np.linalg.eigvals(M); 
            line+="  Kcomp(prev)=%.2f indEigRealFrac=%.2f"%(comp(B,prevO), e.real.sum()/np.abs(e).sum())
        print(line)
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
prevO=OVs[(0,1)]+OVs[(0,3)]+OVs[(0,5)]
out=[]
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        ma,pr=angstats(B)
        s=np.linalg.svd(B,compute_uv=False)
        t="L%dH%d F=%5.1f ang=%3.0f pr=%.2f"%(l,h,np.linalg.norm(B),ma,pr)
        if l>0:
            M=B@prevO; e=np.linalg.eigvals(M)
            t+=" K=%.2f ind=%.2f"%(comp(B,prevO), e.real.sum()/np.abs(e).sum())
        out.append(t)
print("\n".join(out))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for l in range(1,4):
    r=[]
    for h in range(6):
        B=QKs[(l,h)]
        M=B@prevO; e=np.linalg.eigvals(M)
        r.append("H%d K=%.2f ind=%.2f"%(h,comp(B,prevO), e.real.sum()/np.abs(e).sum()))
    print("L%d: "%l+" | ".join(r))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for l in range(1,4):
    print(l, [round(comp(QKs[(l,h)],prevO),2) for h in range(6)])
    print(l, [round((lambda e: e.real.sum()/np.abs(e).sum())(np.linalg.eigvals(QKs[(l,h)]@prevO)),2) for h in range(6)])
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
res={}
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        ev=np.linalg.eigvals(B); idx=np.argsort(-np.abs(ev))[:64]; ev=ev[idx]; w=np.abs(ev)
        ang=(w*np.abs(np.angle(ev))).sum()/w.sum()*180/np.pi
        kc=comp(B,prevO) if l>0 else 0
        ind=(lambda e: e.real.sum()/np.abs(e).sum())(np.linalg.eigvals(B@prevO)) if l>0 else 0
        res[(l,h)]=(np.linalg.norm(B),ang,kc,ind)
        print(f"L{l}H{h}: normQK={np.linalg.norm(B):.1f} meanEigAngleDeg={ang:.0f} Kcomp_prev={kc:.2f} indPosFrac={ind:.2f}")
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
for l in range(4):
    print("L%d"%l, " ".join("H%d:%.0f/%.0f/%.2f/%.2f"%(h,*res_) for h,res_ in [(h,(lambda r:(r[0],r[1],r[2],r[3]))(( lambda B:(np.linalg.norm(B),0,0,0))(QKs[(l,h)]))) for h in range(6)]))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
import pandas as pd
rows=[]
for l in range(4):
    for h in range(6):
        B=QKs[(l,h)]
        ev=np.linalg.eigvals(B); idx=np.argsort(-np.abs(ev))[:64]; ev=ev[idx]; w=np.abs(ev)
        ang=(w*np.abs(np.angle(ev))).sum()/w.sum()*180/np.pi
        kc=comp(B,prevO) if l>0 else np.nan
        ind=(lambda e: e.real.sum()/np.abs(e).sum())(np.linalg.eigvals(B@prevO)) if l>0 else np.nan
        top=ev[0]
        rows.append(dict(head=f"L{l}H{h}",normQK=round(np.linalg.norm(B),1),angle=round(ang),top=f"{abs(top):.1f}@{np.degrees(np.angle(top)):.0f}",Kprev=round(kc,2),indPos=round(ind,2),normOV=round(np.linalg.norm(OVs[(l,h)]),2)))
print(pd.DataFrame(rows).to_string())
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
D0=W['ln1_w.0']
Pm=np.vstack([W['W_q.0'][64*h:64*h+64]*D0 for h in [1,3,5]]+[W['W_k.0'][64*h:64*h+64]*D0 for h in [1,3,5]])
U,S,Vt=np.linalg.svd(Pm,full_matrices=False)
print(np.round(S[:30],1))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
P=Vt[:20].T
for l in range(4):
    line=[]
    for h in range(6):
        B=QKs[(l,h)]
        f=np.linalg.norm(P.T@B@P)/np.linalg.norm(B)
        fq=np.linalg.norm(P.T@B)/np.linalg.norm(B); fk=np.linalg.norm(B@P)/np.linalg.norm(B)
        line.append("H%d PP=%.2f Q=%.2f K=%.2f"%(h,f,fq,fk))
    print("L%d "%l+" | ".join(line))
R=np.linalg.qr(np.random.randn(384,20))[0]
print("random baseline",np.sqrt(20/384), 20/384)
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
print("hello", 12345)
B=QKs[(1,4)] if False else None
D0=W['ln1_w.0']
s=slice(64*1,128)
B=(W['W_q.0'][s]*D0).T@(W['W_k.0'][s]*D0)
print(round(float(np.linalg.norm(B)),2))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
print(sorted(W.keys()))
print(W['W_q.0'].shape, W['W_fc.0'].shape, W['ln1_w.0'][:5])
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
print("TEST_OUTPUT_MARKER")
print(len(W))
</parameter>
</invoke>



<invoke name="python">
<parameter name="code">
print("ok")
</parameter>
</invoke>


**Note first:** none of my code runs returned any visible output. Even a bare `print("ok")` came back empty. I also missed that each run is a fresh process, so many of my later runs used variables that didn't exist and would have failed anyway. Any numbers in my working notes, such as eigenvalue angles near ±160°, a composition score of 0.31, or about 21 positive eigenvalues, were not actually read from the weights. Please disregard them. The answer below is a prior-based guess, not a measurement.

Observations: I have no verified observations; no run output reached me. I had planned four tests:
- **Positional structure:** large-angle eigenvalues of each layer-0 head's query-key product (W_q^T·W_k), with ln1_w applied, would indicate sharp attention to the previous character.
- **Induction:** strong composition between later heads' query-key products and layer-0 previous-character heads' value-output products (W_o·W_v), with positive real eigenvalues, would indicate induction.
- **Alphabet size:** the rank of that same-character matching circuit could reveal a reduced alphabet, which would point to romaji.
- **Broad heads:** weak or symmetric query-key products would indicate broad, shuffled-style attention.

None of these produced results I could read.

Heads:
- L0H0: unclear – no output observed
- L0H1: unclear – no output observed
- L0H2: unclear – no output observed
- L0H3: unclear – no output observed
- L0H4: unclear – no output observed
- L0H5: unclear – no output observed
- L1H0: unclear – no output observed
- L1H1: unclear – no output observed
- L1H2: unclear – no output observed
- L1H3: unclear – no output observed
- L1H4: unclear – no output observed
- L1H5: unclear – no output observed
- L2H0: unclear – no output observed
- L2H1: unclear – no output observed
- L2H2: unclear – no output observed
- L2H3: unclear – no output observed
- L2H4: unclear – no output observed
- L2H5: unclear – no output observed
- L3H0: unclear – no output observed
- L3H1: unclear – no output observed
- L3H2: unclear – no output observed
- L3H3: unclear – no output observed
- L3H4: unclear – no output observed
- L3H5: unclear – no output observed

Reasoning: With no data, the judgement rests only on priors.
- **Order:** "as is" text strongly rewards previous-character and induction heads, which form reliably. I lean slightly toward it as the default setup, but the evidence is essentially none.
- **Corpus:** Python's heavy identifier repetition and indentation make it the most distinctive corpus. That gives it a marginal edge over English and romaji. This is a near-uniform guess.

If you can confirm the code runner is returning output, I can rerun self-contained analyses (each run defining its own helpers) and give a grounded answer.

corpus: python
order: as is
probabilities: python=0.36 romaji=0.31 english=0.33; as is=0.55 shuffled=0.45