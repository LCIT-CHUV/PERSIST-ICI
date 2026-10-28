import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit
from scipy.stats import mannwhitneyu, chi2, rankdata, spearmanr, ttest_1samp
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


def mw_auc(x, y, exact=True):
    x = np.asarray(pd.Series(x).dropna(), float); y = np.asarray(pd.Series(y).dropna(), float)
    if not len(x) or not len(y): return {'n1':len(x),'n0':len(y),'auc':np.nan,'p':np.nan,'u':np.nan}
    ties = len(np.unique(np.r_[x,y])) < len(x)+len(y)
    method = 'exact' if exact and not ties else 'asymptotic'
    r = mannwhitneyu(x, y, alternative='two-sided', method=method)
    return {'n1':len(x),'n0':len(y),'auc':float(r.statistic/(len(x)*len(y))), 'p':float(r.pvalue),'u':float(r.statistic),'method':method}

def exact_mw_even_with_ties(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    r=mannwhitneyu(x,y,alternative='two-sided',method='exact')
    return {'auc':float(r.statistic/(len(x)*len(y))), 'p':float(r.pvalue),'u':float(r.statistic)}

def bh(pvals):
    arr=np.asarray(pvals,float); return multipletests(arr, method='fdr_bh')[1]

def bootstrap_auc(x,y,n=100000,seed=20260916):
    x=np.asarray(x,float); y=np.asarray(y,float); pooled=np.r_[x,y]; labels=np.r_[np.ones(len(x),int),np.zeros(len(y),int)]
    rng=np.random.default_rng(seed); vals=[]; N=len(pooled); done=0; batch=250
    while done<n:
        k=min(batch,n-done); idx=rng.integers(0,N,size=(k,N)); v=pooled[idx]; lab=labels[idx]
        n1=lab.sum(axis=1); n0=N-n1; ok=(n1>0)&(n0>0)
        if np.any(ok):
            ranks=rankdata(v[ok],axis=1); labs=lab[ok]; n1o=n1[ok]; n0o=n0[ok]
            u=(ranks*labs).sum(axis=1)-n1o*(n1o+1)/2
            auc=u/(n1o*n0o); vals.extend(auc.tolist())
        done+=k
    return np.quantile(vals,[.025,.975]).tolist()

def exact_label_permutation_median_p(values, event_mask):
    values=np.asarray(values,float); event_mask=np.asarray(event_mask,bool); k=event_mask.sum(); obs=np.median(values[event_mask])-np.median(values[~event_mask]); total=0; extreme=0
    for comb in itertools.combinations(range(len(values)), k):
        m=np.zeros(len(values),bool); m[list(comb)]=True
        d=np.median(values[m])-np.median(values[~m]); total+=1
        if abs(d)>=abs(obs)-1e-15: extreme+=1
    return obs, extreme/total

def spearman_permutation(x,y,n=100000,seed=20261001):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    obs=float(spearmanr(x,y).statistic); rx=rankdata(x); ry=rankdata(y); rx=(rx-rx.mean())/rx.std(ddof=0); ry=(ry-ry.mean())/ry.std(ddof=0); rng=np.random.default_rng(seed); ext=0
    batch=5000; done=0
    while done<n:
        k=min(batch,n-done); perms=np.empty((k,len(ry)),float)
        for i in range(k): perms[i]=rng.permutation(ry)
        rhos=(perms*rx).mean(axis=1); ext += int(np.count_nonzero(np.abs(rhos)>=abs(obs)-1e-15)); done += k
    return obs, ext/n

def bootstrap_spearman_ci(x,y,n=10000,seed=20261001):
    x=np.asarray(x,float); y=np.asarray(y,float); rng=np.random.default_rng(seed); vals=[]
    batch=5000; done=0; N=len(x)
    while done<n:
        k=min(batch,n-done); idx=rng.integers(0,N,size=(k,N)); xx=x[idx]; yy=y[idx]
        rx=rankdata(xx,axis=1); ry=rankdata(yy,axis=1)
        rx=rx-rx.mean(axis=1,keepdims=True); ry=ry-ry.mean(axis=1,keepdims=True)
        den=np.sqrt((rx*rx).sum(axis=1)*(ry*ry).sum(axis=1)); ok=den>0
        corr=np.empty(k); corr[:]=np.nan; corr[ok]=(rx[ok]*ry[ok]).sum(axis=1)/den[ok]
        vals.extend(corr[np.isfinite(corr)].tolist()); done+=k
    return np.quantile(vals,[.025,.975]).tolist()

def geometric_mean_ratio_and_t(log2_ratios):
    z=np.asarray(log2_ratios,float); t=ttest_1samp(z,0); mean=z.mean(); se=z.std(ddof=1)/np.sqrt(len(z)); from scipy.stats import t as tdist
    q=tdist.ppf(.975,len(z)-1); ci=(2**(mean-q*se),2**(mean+q*se)); return 2**mean,ci,float(t.pvalue)

def cox_ph(time,status,x,scale=1.0):
    X=(np.asarray(x,float)/scale)[:,None]; res=PHReg(np.asarray(time,float),X,status=np.asarray(status,int),ties='efron').fit(); ci=res.conf_int()[0]
    return {'hr':float(np.exp(res.params[0])),'ci_low':float(np.exp(ci[0])),'ci_high':float(np.exp(ci[1])),'p':float(res.pvalues[0]),'beta':float(res.params[0])}

# Firth logistic with Jeffreys-prior penalty
def firth_loglik(beta,X,y):
    eta=X@beta; p=expit(eta); w=p*(1-p); ll=np.sum(y*eta-np.logaddexp(0,eta)); info=X.T@(w[:,None]*X); sign,ld=np.linalg.slogdet(info)
    if sign<=0 or not np.isfinite(ld): return -1e100
    return float(ll+0.5*ld)
def firth_fit(X,y,start=None):
    X=np.asarray(X,float); y=np.asarray(y,float); start=np.zeros(X.shape[1]) if start is None else np.asarray(start,float)
    f=lambda b:-firth_loglik(b,X,y); r=minimize(f,start,method='BFGS',options={'gtol':1e-9,'maxiter':2000})
    if not r.success: r=minimize(f,r.x,method='Powell',options={'maxiter':5000,'xtol':1e-11,'ftol':1e-11})
    return np.asarray(r.x,float), float(-r.fun)
def _profile_ll(j,val,X,y,full):
    idx=[k for k in range(X.shape[1]) if k!=j]
    def f(th):
        b=full.copy(); b[j]=val; b[idx]=th; return -firth_loglik(b,X,y)
    r=minimize(f,full[idx],method='BFGS',options={'gtol':1e-9,'maxiter':2000})
    if not r.success: r=minimize(f,r.x,method='Powell',options={'maxiter':5000,'xtol':1e-11,'ftol':1e-11})
    return float(-r.fun)
def firth_profile_parameter(X,y,j=1):
    X=np.asarray(X,float); y=np.asarray(y,float); b,ll=firth_fit(X,y); ll0=_profile_ll(j,0,X,y,b); p=float(chi2.sf(2*(ll-ll0),1)); target=chi2.ppf(.95,1); bh=b[j]
    def root(v): return 2*(ll-_profile_ll(j,v,X,y,b))-target
    lo=bh-1; hi=bh+1
    while root(lo)<0 and lo>-30: lo-=1
    while root(hi)<0 and hi<30: hi+=1
    lower=brentq(root,lo,bh); upper=brentq(root,bh,hi)
    return {'beta':float(bh),'or':float(np.exp(bh)),'p':p,'ci_low':float(np.exp(lower)),'ci_high':float(np.exp(upper)),'coef_all':b.tolist()}
