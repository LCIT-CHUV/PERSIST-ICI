import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from .io import clean_long, merge_outcomes
from .stats import mw_auc, bootstrap_auc, cox_ph, geometric_mean_ratio_and_t
from scipy.stats import wilcoxon

def _icos_long(master,cohort):
    x=clean_long(master['mmi']); x=x[x.Cell_Type.eq('CD4_EM_ICOS')].copy(); return merge_outcomes(x,cohort)

def initial_gmm_point(master,cohort,seed=20260916,n_init=50):
    m=_icos_long(master,cohort); z=m[m.Timepoint.eq('Initial irAE 1')].dropna(subset=['Value']).copy(); vals=z.Value.to_numpy(float)[:,None]
    gm=GaussianMixture(n_components=2,random_state=seed,n_init=n_init).fit(vals); lab=gm.predict(vals); means=gm.means_.ravel(); lo=int(np.argmin(means)); hi=int(np.argmax(means)); low=vals.ravel()[lab==lo]; high=vals.ravel()[lab==hi]; threshold=(low.max()+high.min())/2
    return {'n':len(z),'low_n':len(low),'high_n':len(high),'mean_low':float(means[lo]),'mean_high':float(means[hi]),'highest_low':float(low.max()),'lowest_high':float(high.min()),'threshold':float(threshold)}
def initial_gmm(master,cohort,seed=20260916,n_init=50,bootstrap=300):
    m=_icos_long(master,cohort); z=m[m.Timepoint.eq('Initial irAE 1')].dropna(subset=['Value']).copy(); vals=z.Value.to_numpy(float)[:,None]
    gm=GaussianMixture(n_components=2,random_state=seed,n_init=n_init).fit(vals); lab=gm.predict(vals); means=gm.means_.ravel(); lo=int(np.argmin(means)); hi=int(np.argmax(means)); low=vals.ravel()[lab==lo]; high=vals.ravel()[lab==hi]; threshold=(low.max()+high.min())/2
    rng=np.random.default_rng(seed); aris=[]
    for _ in range(bootstrap):
        idx=rng.integers(0,len(vals),len(vals)); vv=vals[idx]; g=GaussianMixture(n_components=2,random_state=seed,n_init=1).fit(vv); pred=g.predict(vals); # align not needed for ARI
        aris.append(adjusted_rand_score(lab,pred))
    loo=[]
    for i in range(len(vals)):
        g=GaussianMixture(n_components=2,random_state=seed,n_init=min(n_init,5)).fit(np.delete(vals,i,axis=0)); pred=g.predict(vals[i:i+1])[0]; ml=g.means_.ravel(); plow=int(np.argmin(ml)); classification='low' if pred==plow else 'high'; full='low' if lab[i]==lo else 'high'; loo.append(classification==full)
    return {'n':len(z),'low_n':len(low),'high_n':len(high),'mean_low':float(means[lo]),'mean_high':float(means[hi]),'highest_low':float(low.max()),'lowest_high':float(high.min()),'threshold':float(threshold),'bootstrap_ari_mean':float(np.mean(aris)),'loo_agreement':float(np.mean(loo))}
def last_pre_d0(master,cohort):
    m=_icos_long(master,cohort); m=m[m.Date<m.d0].sort_values(['PatientID','Date']).groupby('PatientID').tail(1); a=m.loc[m.event,'Value']; b=m.loc[~m.event,'Value']; st=mw_auc(a,b,exact=False); return m, {'n':len(m),'events':int(m.event.sum()),'event_median':float(a.median()),'control_median':float(b.median()),**st,'median_days_before_d0':float((m.d0-m.Date).dt.days.median()),'range_days_before_d0':[int((m.d0-m.Date).dt.days.min()),int((m.d0-m.Date).dt.days.max())]}
def rechallenge1(master,cohort,bootstrap_draws=100000,seed=20260916):
    m=_icos_long(master,cohort); m=m[m.Timepoint.eq('Rechallenge 1')].copy(); m=m[(m.Date>=m.d0)&(~m.event|(m.Date<m.event_date))].copy(); a=m.loc[m.event,'Value']; b=m.loc[~m.event,'Value']; st=mw_auc(a,b,exact=False); ci=bootstrap_auc(a,b,n=bootstrap_draws,seed=seed) if bootstrap_draws and bootstrap_draws>0 else None; return m, {'n':len(m),'events':int(m.event.sum()),'event_median':float(a.median()),'control_median':float(b.median()),**st,'bootstrap_auc_ci':ci,'median_day':float((m.Date-m.d0).dt.days.median()),'range_day':[int((m.Date-m.d0).dt.days.min()),int((m.Date-m.d0).dt.days.max())]}
def cox_from_r1(r1,censor='2026-01-16',scale=2.31):
    censor=pd.Timestamp(censor); end=r1.event_date.where(r1.event,censor); time=(end-r1.Date).dt.days; return cox_ph(time,r1.event.astype(int),r1.Value,scale=scale)
def event_pairs(master,cohort):
    m=_icos_long(master,cohort); pre=m[m.Date<m.d0].sort_values(['PatientID','Date']).groupby('PatientID').tail(1); ev=m[m.event & m.Timepoint.astype(str).str.startswith('Rechallenge irAE') & (m.Date>=m.event_date)].sort_values(['PatientID','Date']).groupby('PatientID').head(1); p=pre.merge(ev,on='PatientID',suffixes=('_pre','_event')); same=p[p['B or T-cell panel_pre'].eq(p['B or T-cell panel_event'])].copy(); same['log2_ratio']=np.log2(same.Value_event/same.Value_pre); gmr,ci,pt=geometric_mean_ratio_and_t(same.log2_ratio); ps=float(wilcoxon(same.log2_ratio,alternative='two-sided',method='exact').pvalue); return same, {'n':len(same),'increases':int((same.Value_event>same.Value_pre).sum()),'decreases':int((same.Value_event<same.Value_pre).sum()),'median_pre':float(same.Value_pre.median()),'median_event':float(same.Value_event.median()),'gmr':gmr,'ci_low':ci[0],'ci_high':ci[1],'p_t':pt,'p_wilcoxon':ps}
