"""Focused soluble analysis.

Important provenance note: the final focused Table S14 cohort cannot be reconstructed solely from the V15 raw Cytokines sheet.
This module is operational on a curated patient-level CSV if supplied. It deliberately refuses to claim Table-S14 reproduction from raw V15 alone.
"""
import pandas as pd, numpy as np
from .stats import exact_mw_even_with_ties
REQUIRED=['patient_id','marker','window','value','event','sample_date','event_date']
def run_curated(path):
    d=pd.read_csv(path); missing=[c for c in REQUIRED if c not in d.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    d['sample_date']=pd.to_datetime(d.sample_date); d['event_date']=pd.to_datetime(d.event_date,errors='coerce'); d['event']=d.event.astype(bool)
    out={}
    for (marker,window),g in d.groupby(['marker','window']):
        a=g.loc[g.event,'value']; b=g.loc[~g.event,'value']; st=exact_mw_even_with_ties(a,b); out[f'{marker}:{window}']={'events':len(a),'controls':len(b),'event_median':float(a.median()),'control_median':float(b.median()),**st}
    return out
def raw_v15_diagnostic(master,cohort):
    d=master['cytokines'].copy(); d['DATE']=pd.to_datetime(d.DATE,errors='coerce'); d['VALEUR_CLEAN']=pd.to_numeric(d.VALEUR_CLEAN,errors='coerce'); d=d.merge(cohort[['patient_id','d0','event','event_date']],left_on='ID',right_on='patient_id'); d['day']=(d.DATE-d.d0).dt.days; out={}
    for marker in ['sCD25','CXCL12']:
        for window,lo,hi in [('PreR',-30,-1),('Early',0,30)]:
            g=d[(d.TYPE==marker)&(d.day>=lo)&(d.day<=hi)&(~d.event|(d.DATE<d.event_date))].groupby('ID',as_index=False).agg(value=('VALEUR_CLEAN','median'),event=('event','first'))
            if len(g):
                a=g.loc[g.event,'value']; b=g.loc[~g.event,'value']; st=exact_mw_even_with_ties(a,b) if len(a) and len(b) else {}; out[f'{marker}:{window}']={'events':len(a),'controls':len(b),'event_median':float(a.median()) if len(a) else None,'control_median':float(b.median()) if len(b) else None,**st}
    out['warning']='Raw V15 Cytokines does not reproduce the final curated Table S14 cohort; do not substitute these diagnostics for the locked focused-soluble results.'
    return out
