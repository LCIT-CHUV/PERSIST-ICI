import math, json, yaml
from pathlib import Path

def close(a,b,atol): return a is not None and math.isfinite(float(a)) and abs(float(a)-float(b))<=atol
def load_expected(path): return yaml.safe_load(Path(path).read_text())
def check(actual, expected, tolerances=None):
    tolerances=tolerances or {}; rows=[]
    def walk(a,e,p=''):
        for k,v in e.items():
            key=f'{p}.{k}' if p else k
            if isinstance(v,dict): walk(a.get(k,{}) if isinstance(a,dict) else {},v,key)
            else:
                av=a.get(k) if isinstance(a,dict) else None; tol=tolerances.get(key, 1e-6 if isinstance(v,(int,float)) else None)
                if isinstance(v,(int,float)): ok=av is not None and close(av,v,tol)
                else: ok=av==v
                rows.append({'key':key,'expected':v,'actual':av,'tolerance':tol,'pass':bool(ok)})
    walk(actual,expected); return rows
