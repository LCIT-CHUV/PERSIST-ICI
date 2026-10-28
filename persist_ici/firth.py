import numpy as np, pandas as pd
from .stats import firth_profile_parameter

def run(r1):
    x=r1.copy(); x['icos_iqr']=x.Value.astype(float)/2.31; x['T_panel']=(x['B or T-cell panel']=='T').astype(float); x['multi']=x.multiorgan_irAE.astype(str).str.lower().eq('y').astype(float); x['anti_il6']=x.prophylaxis_class_primary.astype(str).str.contains('IL-6',case=False,na=False).astype(float); x['grade3']=(pd.to_numeric(x.initial_irAE_max_grade,errors='coerce')>=3).astype(float); x['dual']=x.dual_ici_exposure.astype(str).str.lower().eq('y').astype(float); x['gc']=x.OnGC.astype(str).str.lower().eq('yes').astype(float); x['days']=(x.Date-x.d0).dt.days.astype(float); y=x.event.astype(int).to_numpy()
    out={}
    for name,c in [('T-panel availability','T_panel'),('Multi-organ initial irAE','multi'),('Anti–IL-6R prophylaxis','anti_il6'),('Initial grade ≥3','grade3'),('Prior dual ICI','dual'),('Active glucocorticoids','gc'),('Days from D0','days')]:
        X=np.column_stack([np.ones(len(x)),x.icos_iqr.to_numpy(float),x[c].to_numpy(float)]); out[name]=firth_profile_parameter(X,y,j=1)
    return out
