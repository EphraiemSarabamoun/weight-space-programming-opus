```python
for l in range(4):
    fc=W[f'W_fc.{l}']; pj=W[f'W_proj.{l}']
    rn=np.linalg.norm(fc*W[f'ln2_w.{l}'],axis=1); cn=np.linalg.norm(pj,axis=0)
    prod=rn*cn
    print(l,'fc row norm pct',np.round(np.percentile(rn,[1,5,25,50,75,95,99]),2),'proj col pct',np.round(np.percentile(cn,[1,5,25,50,75,95,99]),2))
    print('   #proj col <0.3:',(cn<0.3).sum(),' <0.5:',(cn<0.5).sum(),' prod pct',np.round(np.percentile(prod,[1,5,25,50,75,95]),2))
```