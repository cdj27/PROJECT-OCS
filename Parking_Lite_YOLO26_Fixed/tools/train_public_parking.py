"""Download labelled photos, fine-tune, evaluate and retain only a better candidate."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def is_better(baseline,candidate):
    """No measured regression in occupancy safety or bay coverage/precision."""
    higher=('bay_recall','bay_precision','occupied_recall','available_precision')
    no_regression=all(candidate[k]>=baseline[k]-1e-9 for k in higher)
    no_regression &= candidate['false_available_rate']<=baseline['false_available_rate']+1e-9
    improvement=any(candidate[k]>baseline[k]+1e-9 for k in higher)
    improvement |= candidate['false_available_rate']<baseline['false_available_rate']-1e-9
    return bool(no_regression and improvement)


def write_comparison(output,candidate_path,activate=False):
    output=Path(output)
    reports={name:json.loads((output/f'{name}.json').read_text()) for name in
             ('baseline-val','candidate-val','baseline-test','candidate-test')}
    validation_improved=is_better(reports['baseline-val']['metrics'],reports['candidate-val']['metrics'])
    b=reports['baseline-test']['metrics'];c=reports['candidate-test']['metrics']
    test_safe=all(c[k]>=b[k]-1e-9 for k in ('bay_recall','bay_precision','occupied_recall','available_precision')) and c['false_available_rate']<=b['false_available_rate']+1e-9
    candidate_path=Path(candidate_path).resolve()
    deploy=bool(activate and validation_improved and test_safe)
    if deploy:
        live=ROOT/'models'/'parking_slots.pt'
        if live.exists():
            backup=ROOT/'models'/'backups';backup.mkdir(exist_ok=True)
            shutil.copy2(live,backup/f'parking-slots-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.pt')
        staging=live.with_suffix('.incoming.pt');shutil.copy2(candidate_path,staging);staging.replace(live)
    summary=('Fine-tuned candidate activated; restart the server to load it.' if deploy else
             'Fine-tuning completed. The current model is retained because the candidate did not pass the improvement gate.'
             if not (validation_improved and test_safe) else
             'Fine-tuning completed. Candidate passed evaluation; activation was not requested.')
    result={'summary':summary,'candidate':str(candidate_path),'activated':deploy,
            'validation_improved':validation_improved,'test_no_regression':bool(test_safe),
            'metrics':{name:report['metrics'] for name,report in reports.items()},
            'note':'This is a small PKLot pilot. Parent pretraining may overlap PKLot; results do not establish accuracy on arbitrary uploaded photos.'}
    (output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--activate-if-better',action='store_true')
    a=p.parse_args()
    name='public-pklot-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'outputs'/name;output.mkdir(parents=True)
    dataset=ROOT/'datasets'/f'public_pklot_{a.count}'
    baseline=ROOT/'models'/'parking_slots.pt'
    if not baseline.exists():baseline=ROOT/'models'/'parking_slots.evaluation.pt'
    if not baseline.exists():p.error('Install the parking-specific baseline weights first')
    def run(script,args,label):
        print(f'{label} (log: {output/(label+".log")})',flush=True)
        with (output/(label+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools'/script),*map(str,args)],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    run('import_public_parking.py',['--count',a.count,'--output',dataset],'import')
    data=dataset/'data.yaml'
    run('benchmark_parking.py',['--data',data,'--model',baseline,'--split','val','--output',output/'baseline-val.json'],'baseline-val')
    run('train_parking.py',['--task','slots','--data',data,'--model',baseline,'--epochs',a.epochs,
        '--imgsz',512,'--batch',2,'--freeze',23,'--lr0',.0001,'--name',name],'training')
    candidate=ROOT/'runs'/'parking'/name/'weights'/'best.pt'
    for split in ('val','test'):
        run('benchmark_parking.py',['--data',data,'--model',candidate,'--split',split,'--output',output/f'candidate-{split}.json'],f'candidate-{split}')
    run('benchmark_parking.py',['--data',data,'--model',baseline,'--split','test','--output',output/'baseline-test.json'],'baseline-test')
    result=write_comparison(output,candidate,a.activate_if_better)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
