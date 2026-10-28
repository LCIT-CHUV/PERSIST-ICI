import numpy as np, pandas as pd
from .io import clean_long, merge_outcomes
from .stats import mw_auc, exact_label_permutation_median_p

def slopes(master,cohort,marker='CD4_LAG3',max_day=None,exclude=('Prophy2n-033',)):
    d=clean_long(master['mmi_log2fc']); d=d[(d['B or T-cell panel']=='T')&d.Cell_Type.eq(marker)].copy(); d=merge_outcomes(d,cohort); d['day']=(d.Date-d.d0).dt.days; d=d[(d.day>0)&(~d.event|(d.Date<d.event_date))&(~d.PatientID.isin(exclude))];
    if max_day is not None: d=d[d.day<=max_day]
    rows=[]
    for pid,g in d.groupby('PatientID'):
        if len(g)<2: continue
        slope=float(np.polyfit(g.day.to_numpy(float),g.Value.to_numpy(float),1)[0]*30); rows.append({'PatientID':pid,'n_obs':len(g),'event':bool(g.event.iloc[0]),'slope':slope,'min_day':int(g.day.min()),'max_day':int(g.day.max())})
    z=pd.DataFrame(rows); a=z.loc[z.event,'slope']; b=z.loc[~z.event,'slope']; st=mw_auc(a,b,exact=True); obs,pp=exact_label_permutation_median_p(z.slope,z.event)
    return z, {'n':len(z),'events':int(z.event.sum()),'event_median':float(a.median()),'control_median':float(b.median()),'median_difference':float(obs),'patient_p':st['p'],'auc':st['auc'],'permutation_median_p':float(pp),'observations':int(d[d.PatientID.isin(z.PatientID)].shape[0]),'two_observation_patients':int((z.n_obs==2).sum())}
def leave_one_patient_out(z):
    ds=[]
    for pid in z.PatientID:
        q=z[z.PatientID!=pid]; ds.append(float(q.loc[q.event,'slope'].median()-q.loc[~q.event,'slope'].median()))
    return {'n':len(ds),'positive':int(np.sum(np.asarray(ds)>0)),'min':float(np.min(ds)),'max':float(np.max(ds))}
