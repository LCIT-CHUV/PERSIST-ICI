"""Documented broad-screen reconstruction utilities.

The exact historical 935-test universe depended on historical feature-block inclusion/coverage definitions that are not fully preserved in the final artifacts. This module reproduces the documented temporal aggregation, preprocessing, continuous testing and multiplicity logic, but it is not labelled as an exact 935-test reproduction unless the historical feature manifest is supplied.
"""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import RobustScaler
from .stats import bh

def window_patient_medians(df, cohort, id_col, date_col, feature_col, value_col, lo, hi, anchor='d0', require_pre_event=True):
    x=df.copy(); x[date_col]=pd.to_datetime(x[date_col],errors='coerce'); x=x.merge(cohort[['patient_id',anchor,'event','event_date']],left_on=id_col,right_on='patient_id'); x['rel_day']=(x[date_col]-x[anchor]).dt.days; x=x[(x.rel_day>=lo)&(x.rel_day<=hi)];
    if require_pre_event: x=x[(~x.event)|(x[date_col]<x.event_date)]
    return x.groupby([id_col,feature_col],as_index=False)[value_col].median()
def preprocess_matrix(wide):
    x=wide.copy().astype(float)
    for c in x.columns:
        s=x[c].dropna()
        if len(s) and (s>=0).all() and abs(s.skew())>1: x[c]=np.log1p(x[c])
        med=x[c].median(); x[c]=x[c].fillna(med); lo,hi=x[c].quantile([.025,.975]); x[c]=x[c].clip(lo,hi)
    return pd.DataFrame(RobustScaler().fit_transform(x),index=x.index,columns=x.columns)
def continuous_tests(patient_feature, outcomes, id_col='ID', feature_col='TYPE', value_col='VALEUR_CLEAN'):
    d=patient_feature.merge(outcomes[['patient_id','event']],left_on=id_col,right_on='patient_id'); rows=[]
    for feat,g in d.groupby(feature_col):
        a=g.loc[g.event,value_col].dropna(); b=g.loc[~g.event,value_col].dropna()
        if len(a)<2 or len(b)<2: continue
        r=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic'); rows.append({'feature':feat,'n_event':len(a),'n_control':len(b),'p':r.pvalue,'auc':r.statistic/(len(a)*len(b))})
    z=pd.DataFrame(rows)
    if len(z): z['q_bh']=bh(z.p)
    return z
