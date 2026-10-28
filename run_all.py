#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
from persist_ici.io import load_master, clean_cohort
from persist_ici import clinical, icos, firth, lag3, treg, monocytes, cd95, soluble

def log(msg):
    print(f"[PERSIST-ICI] {msg}", file=sys.stderr, flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master-file',required=True)
    ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--soluble-curated',default=None)
    ap.add_argument('--fast',action='store_true',help='Use fewer ICOS bootstrap draws for developer checks.')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    t0=time.time(); log('loading master workbook')
    master=load_master(args.master_file); cohort=clean_cohort(master['cohort']); results={}
    log('clinical'); results['clinical']=clinical.run(cohort)
    log('ICOS pre-D0'); pre,results['icos_pred0']=icos.last_pre_d0(master,cohort)
    log('ICOS R1'); r1,results['icos_r1']=icos.rechallenge1(master,cohort,bootstrap_draws=0)
    log('ICOS Cox'); results['icos_cox']=icos.cox_from_r1(r1)
    log('Firth'); results['firth']=firth.run(r1)
    pairs=None
    log('LAG3 full'); lf,results['lag3_full']=lag3.slopes(master,cohort,max_day=None)
    log('LAG3 D0-180'); ld,results['lag3_d180']=lag3.slopes(master,cohort,max_day=180); results['lag3_d180']['loo']=lag3.leave_one_patient_out(ld)
    log('ICOS-Treg paired'); tp,results['treg_paired']=treg.paired(master,cohort)
    log('ICOS-Treg D0-180'); ts,results['treg_d180']=treg.d180_slopes(master,cohort)
    log('monocytes R1'); mr,results['monocytes_r1']=monocytes.r1(master,cohort)
    log('monocytes paired'); mp,results['monocytes_paired']=monocytes.paired(master,cohort,mr)
    log('CD95'); cd,results['cd95']=cd95.run(master,cohort)
    log('soluble diagnostic'); results['soluble_raw_v15_diagnostic']=soluble.raw_v15_diagnostic(master,cohort)
    log('ICOS GMM point characterization'); results['icos_gmm']=icos.initial_gmm_point(master,cohort)
    log('ICOS event pairs'); pairs,results['icos_event_pairs']=icos.event_pairs(master,cohort)
    log('ICOS R1 bootstrap CI (final step)'); from persist_ici.stats import bootstrap_auc; _a=r1.loc[r1.event,'Value']; _b=r1.loc[~r1.event,'Value']; results['icos_r1']['bootstrap_auc_ci']=bootstrap_auc(_a,_b,n=2000 if args.fast else 100000,seed=20260916)
    if args.soluble_curated:
        log('soluble curated'); results['soluble_curated']=soluble.run_curated(args.soluble_curated)
    log('writing JSON/CSV outputs')
    (out/'results.json').write_text(json.dumps(results,indent=2,default=lambda x: float(x) if hasattr(x,'item') else str(x)),encoding='utf-8')
    for name,df in [('icos_pred0',pre),('icos_r1',r1),('icos_pairs',pairs),('lag3_d180',ld),('treg_paired',tp),('treg_d180',ts),('monocytes_r1',mr),('monocytes_paired',mp),('cd95_r1',cd)]:
        df.to_csv(out/f'{name}.csv',index=False)
    log(f'done in {time.time()-t0:.1f}s')
    print(json.dumps({'status':'OK','output':str(out/'results.json')},indent=2), flush=True)
    os._exit(0)

if __name__=='__main__': main()
