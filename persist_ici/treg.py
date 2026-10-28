import numpy as np, pandas as pd
from .io import clean_long, merge_outcomes
from .stats import spearman_permutation, bootstrap_spearman_ci, geometric_mean_ratio_and_t
from scipy.stats import spearmanr

def _sample_table(master,cohort):
    d=clean_long(master['treg']); d=d[(d['B or T-cell panel']=='T') & d.Cell_Type.isin(['of_nontreg_icos_mmi','of_cd4_pos_pct_treg_like'])].copy(); p=d.pivot_table(index=['PatientID','Timepoint','Date','OnGC','OnTCZ'],columns='Cell_Type',values='Value',aggfunc='first').reset_index(); p=merge_outcomes(p,cohort); p['day']=(p.Date-p.d0).dt.days; return p
def paired(master,cohort):
    p=_sample_table(master,cohort); pre=p[p.Date<p.d0].sort_values(['PatientID','Date']).groupby('PatientID').tail(1); r1=p[p.Timepoint.eq('Rechallenge 1') & (p.Date>=p.d0) & (~p.event|(p.Date<p.event_date))]; q=pre.merge(r1,on='PatientID',suffixes=('_pre','_r1')); cols=['of_nontreg_icos_mmi','of_cd4_pos_pct_treg_like']; q=q[(q[[c+'_pre' for c in cols]+[c+'_r1' for c in cols]]>0).all(axis=1)].copy(); q['icos_log2']=np.log2(q.of_nontreg_icos_mmi_r1/q.of_nontreg_icos_mmi_pre); q['treg_log2']=np.log2(q.of_cd4_pos_pct_treg_like_r1/q.of_cd4_pos_pct_treg_like_pre); rho,pp=spearman_permutation(q.icos_log2,q.treg_log2,n=100000,seed=20261001); ci=bootstrap_spearman_ci(q.icos_log2.to_numpy(),q.treg_log2.to_numpy(),n=10000,seed=20261001); gmi,cii,pi=geometric_mean_ratio_and_t(q.icos_log2); gmt,cit,pt=geometric_mean_ratio_and_t(q.treg_log2); return q, {'n':len(q),'rho':rho,'permutation_p':pp,'bootstrap_ci':ci,'icos_pre_median':float(q.of_nontreg_icos_mmi_pre.median()),'icos_r1_median':float(q.of_nontreg_icos_mmi_r1.median()),'treg_pre_median':float(q.of_cd4_pos_pct_treg_like_pre.median()),'treg_r1_median':float(q.of_cd4_pos_pct_treg_like_r1.median()),'icos_gmr':gmi,'icos_gmr_ci':cii,'icos_ttest_p':pi,'treg_gmr':gmt,'treg_gmr_ci':cit,'treg_ttest_p':pt}
def d180_slopes(master,cohort):
    p=_sample_table(master,cohort); p=p[(p.day>0)&(p.day<=180)&(~p.event|(p.Date<p.event_date))&~p.PatientID.eq('Prophy2n-033')]; rows=[]
    for pid,g in p.groupby('PatientID'):
        g=g.dropna(subset=['of_nontreg_icos_mmi','of_cd4_pos_pct_treg_like']);
        if len(g)<2 or not (g[['of_nontreg_icos_mmi','of_cd4_pos_pct_treg_like']]>0).all().all(): continue
        a=float(np.polyfit(g.day,np.log2(g.of_nontreg_icos_mmi),1)[0]*30); b=float(np.polyfit(g.day,np.log2(g.of_cd4_pos_pct_treg_like),1)[0]*30); rows.append({'PatientID':pid,'n_obs':len(g),'icos_slope':a,'treg_slope':b})
    z=pd.DataFrame(rows); rho,pp=spearman_permutation(z.icos_slope,z.treg_slope,n=100000,seed=20261001); ci=bootstrap_spearman_ci(z.icos_slope.to_numpy(),z.treg_slope.to_numpy(),n=10000,seed=20261001); return z, {'n':len(z),'observations':int(z.n_obs.sum()),'two_obs_patients':int((z.n_obs==2).sum()),'rho':rho,'permutation_p':pp,'bootstrap_ci':ci}
