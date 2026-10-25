import json, math, numpy as np
from scipy import stats
from itertools import combinations
R=json.load(open('study2_results.json'))
RANKS=[1,2,4,8,16]; LR=[1e-4,3e-4,1e-3,3e-3,1e-2]
cl=lambda a,b: abs(a-b)<=1e-12*max(1,abs(b))
def cell(rk,lr,k): return np.array([r[k] for r in R if r['budget_mode']=='epochs' and r['method']=='lora'
    and r['n_requested']==500 and r['rank']==rk and cl(r['lr'],lr) and cl(r['wd'],0.01)])
MV=np.array([[cell(r,l,'best_val_f1').mean() for l in LR] for r in RANKS])
SV=np.array([[cell(r,l,'best_val_f1').std(ddof=1) for l in LR] for r in RANKS])
MT=np.array([[cell(r,l,'test_acc').mean() for l in LR] for r in RANKS])
ST=np.array([[cell(r,l,'test_acc').std(ddof=1) for l in LR] for r in RANKS])
floorv=np.median(SV); floort=np.median(ST)
SV=np.maximum(SV,floorv); ST=np.maximum(ST,floort)
print(f'median cell sd: val-F1 {floorv:.4f}, test-acc {floort:.4f}')
def shrink(M,s):
    g=M.mean(); rm=M.mean(1,keepdims=True); cm=M.mean(0,keepdims=True)
    add=rm+cm-g; return add+s*(M-add)
def Fint(Y):
    a,b,n=Y.shape; c=Y.mean(2); g=c.mean()
    ssab=n*((c-c.mean(1,keepdims=True)-c.mean(0,keepdims=True)+g)**2).sum()
    sse=((Y-c[...,None])**2).sum(); d1=(a-1)*(b-1); d2=a*b*(n-1)
    return stats.f.sf((ssab/d1)/(sse/d2),d1,d2)
def holm_rev(Y):
    n=Y.shape[2]; T=[]
    for i,j in combinations(range(5),2):
        for l in range(5):
            A,B=Y[i,l],Y[j,l]; va,vb=A.var(ddof=1)/n,B.var(ddof=1)/n; se=math.sqrt(va+vb)
            df=(va+vb)**2/(va**2/(n-1)+vb**2/(n-1))
            for w,s in ((0,1),(1,-1)):
                T.append((stats.t.sf(s*(A.mean()-B.mean())/se,df),(i,j,w)))
    T.sort(); m=len(T); sig=set()
    for k,(p,key) in enumerate(T):
        if p<=0.05/(m-k): sig.add(key)
        else: break
    return any((i,j,0) in sig and (i,j,1) in sig for i,j,_ in sig)
rng=np.random.default_rng(1); NS=2000
print(f'\n{"effect":>8}{"seeds":>6}{"Q1 all":>9}{"Q1 mid,stable":>15}{"Holm rev":>10}{"Q4 material":>13}')
for s in (1.0,0.5):
    mv,mt=shrink(MV,s),shrink(MT,s)
    for n in (3,5,7,10):
        q1=q1m=h=q4=0
        for _ in range(NS):
            Y=mv[...,None]+SV[...,None]*rng.standard_normal((5,5,n))
            Z=mt[...,None]+ST[...,None]*rng.standard_normal((5,5,n))
            q1+=Fint(Y)<.05
            q1m+=Fint(Y[1:4,1:3])<.05
            h+=holm_rev(Y)
            pref=Y.mean(2).argmax(1); tm=Z.mean(2)
            tuned=[tm[i,pref[i]] for i in range(5)]
            fixed=np.median(tm.max(0)-tm.min(0))
            q4+=(max(tuned)-min(tuned))<=0.5*fixed
        print(f'{s:>8}{n:>6}{q1/NS:>9.2f}{q1m/NS:>15.2f}{h/NS:>10.2f}{q4/NS:>13.2f}')
