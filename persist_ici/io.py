from pathlib import Path
import pandas as pd

SHEETS = {
    'cohort': 'COHORT_INFO',
    'routine': 'Routine_Labs',
    'cytokines': 'Cytokines',
    'freq': 'Cytof_Freq',
    'mmi': 'Cytof_MMI',
    'mmi_log2fc': 'Cytof_MMI_log2fc',
    'treg': 'Cytoff_MMI-Freq_Tregs_ICOS',
    'subsets': 'CountsFreqMMI_TregMono_Subsets',
}

def load_master(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    for key, sheet in SHEETS.items():
        header = 1 if sheet == 'COHORT_INFO' else 0
        out[key] = pd.read_excel(path, sheet_name=sheet, header=header)
    return out

def clean_cohort(df):
    x = df.copy()
    for c in ['d0_rechallenge_date','DateIRAErechall_main','initial_irAE_onset_date']:
        if c in x: x[c] = pd.to_datetime(x[c], errors='coerce')
    x['event'] = ~x['irAE_Rechallenge_Main'].astype(str).str.contains('No irAE', case=False, na=False)
    x['event_date'] = x['DateIRAErechall_main']
    x['d0'] = x['d0_rechallenge_date']
    return x

def clean_long(df, date_col='Date'):
    x = df.copy()
    if date_col in x: x[date_col] = pd.to_datetime(x[date_col], errors='coerce')
    if 'Value' in x: x['Value'] = pd.to_numeric(x['Value'], errors='coerce')
    if 'VALEUR_CLEAN' in x: x['VALEUR_CLEAN'] = pd.to_numeric(x['VALEUR_CLEAN'], errors='coerce')
    return x

def merge_outcomes(df, cohort, patient_col='PatientID'):
    cols = ['patient_id','d0','event','event_date','irAE_Rechallenge_Main','dominant_initial_irAE_phenotype',
            'multiorgan_irAE','prophylaxis_class_primary','initial_irAE_max_grade','dual_ici_exposure']
    cols = [c for c in cols if c in cohort]
    return df.merge(cohort[cols], left_on=patient_col, right_on='patient_id', how='left')
