import numpy as np, pandas as pd
from .io import clean_long, merge_outcomes
from .stats import exact_mw_even_with_ties

def r1(master,cohort):
    f=clean_long(master['freq']); f=f[(f['B or T-cell panel']=='T')&f.Timepoint.eq('Rechallenge 1')&f.Cell_Type.isin(['Classical_mono','Intm_mono','non_class_mono'])]; p=f.pivot_table(index=['PatientID','Date','OnGC','OnTCZ'],columns='Cell_Type',values='Value',aggfunc='first').reset_index(); p=merge_outcomes(p,cohort); p=p[(p.Date>=p.d0)&(~p.event|(p.Date<p.event_date))].copy(); p['C_plus_I']=p.Classical_mono+p.Intm_mono; p['sum3mono']=p.C_plus_I+p.non_class_mono
    out={}
    for c in ['Classical_mono','Intm_mono','non_class_mono','C_plus_I','sum3mono']:
        a=p.loc[p.event,c]; b=p.loc[~p.event,c]; st=exact_mw_even_with_ties(a,b); out[c]={'event_median':float(a.median()),'control_median':float(b.median()),**st}
    return p, {'n':len(p),'events':int(p.event.sum()),'readouts':out}
def paired(master,cohort,r1_df):
    f=clean_long(master['freq']); f=f[(f['B or T-cell panel']=='T')&f.Cell_Type.isin(['Classical_mono','Intm_mono','non_class_mono'])]; p=f.pivot_table(index=['PatientID','Date'],columns='Cell_Type',values='Value',aggfunc='first').reset_index(); p=merge_outcomes(p,cohort); pre=p[p.Date<p.d0].sort_values(['PatientID','Date']).groupby('PatientID').tail(1); pre=pre[~pre.PatientID.eq('Prophy2n-074')].copy(); q=pre.merge(r1_df[['PatientID','Date','event','Classical_mono','Intm_mono','non_class_mono']],on='PatientID',suffixes=('_pre','_r1')); q['C_plus_I_pre']=q.Classical_mono_pre+q.Intm_mono_pre; q['C_plus_I_r1']=q.Classical_mono_r1+q.Intm_mono_r1
    out={}
    for c in ['Classical_mono','Intm_mono','non_class_mono','C_plus_I']:
        q[c+'_delta']=q[c+'_r1']-q[c+'_pre']; a=q.loc[q.event_r1,c+'_delta']; b=q.loc[~q.event_r1,c+'_delta']; st=exact_mw_even_with_ties(a,b); out[c]={'event_delta_median':float(a.median()),'control_delta_median':float(b.median()),**st}
    return q, {'n':len(q),'events':int(q.event_r1.sum()),'readouts':out}
