import numpy as np, pandas as pd
from .io import clean_long, merge_outcomes
from .stats import exact_mw_even_with_ties, bh

def run(master,cohort):
    d=clean_long(master['subsets']); d=d[(d['B or T-cell panel']=='T')&d.Timepoint.eq('Rechallenge 1')].copy(); dates=d.groupby('PatientID').Date.min().rename('sample_date'); base=cohort.merge(dates,left_on='patient_id',right_index=True,how='inner'); base=base[(~base.event)|(base.sample_date<base.event_date)].copy(); piv=d[d.PatientID.isin(base.patient_id)].pivot_table(index='PatientID',columns='Cell_Type',values='Value',aggfunc='first').reset_index(); m=base.merge(piv,left_on='patient_id',right_on='PatientID'); rheum=m.irAE_Rechallenge_Main.astype(str).str.contains('Arthritis',case=False,na=False); no=~m.event; other=m.event&~rheum; primary='of_DN_CCR7_pos_NONTREG_CD95'; control='of_DN_CCR7_neg_NONTREG_CD95'; count='ICOS_pos_NONTREG_DN_CCR7_pos_counts'; freq='of_CD4_pos_ICOS_pos_DN_CCR7_pos_NONTREG'
    st=exact_mw_even_with_ties(m.loc[rheum,primary],m.loc[no,primary]); fam=[c for c in m.columns if c.startswith('of_DN_CCR7_pos_NONTREG_')]; ps=[]
    for c in fam: ps.append(exact_mw_even_with_ties(m.loc[rheum,c],m.loc[no,c])['p'])
    qs=bh(ps); idx=fam.index(primary)
    out={'n':len(m),'events':int(m.event.sum()),'rheum_n':int(rheum.sum()),'other_event_n':int(other.sum()),'no_toxicity_n':int(no.sum()),'rheum_median':float(m.loc[rheum,primary].median()),'no_toxicity_median':float(m.loc[no,primary].median()),'other_event_median':float(m.loc[other,primary].median()),'auc':st['auc'],'p':st['p'],'bh_q_43':float(qs[idx]),'family_n':len(fam),'count_rheum_median':float(m.loc[rheum,count].median()),'count_no_toxicity_median':float(m.loc[no,count].median()),'freq_cd4_rheum_median':float(m.loc[rheum,freq].median()),'freq_cd4_no_toxicity_median':float(m.loc[no,freq].median()),'control_ccr7neg_rheum_median':float(m.loc[rheum,control].median()),'control_ccr7neg_no_median':float(m.loc[no,control].median())}
    return m,out
