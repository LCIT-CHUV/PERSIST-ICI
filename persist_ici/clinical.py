import pandas as pd

def run(cohort):
    n=len(cohort); ev=cohort['event'].sum(); grades=[]
    for x in cohort['irAE_Rechallenge_Grades']:
        if pd.isna(x): continue
        toks=str(x).replace(';',',').split(',')
        nums=[]
        for t in toks:
            try: nums.append(float(t.strip()))
            except: pass
        grades.append(max(nums) if nums else None)
    grade3=sum(g is not None and g>=3 for g in grades)
    return {'n':int(n),'recurrent_irae':int(ev),'grade3plus_recurrence':int(grade3)}
