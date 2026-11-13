import json
import pytest
from tools.import_public_parking import iter_samples,choose_splits,annotation_lines
from tools.benchmark_parking import score_image
from tools.train_public_parking import is_better


def test_public_split_holds_out_site_and_capture_days():
    rows=[{'index':i*10000+d*10+j,'site':site,'day':f'2020-01-{d+1:02d}'}
          for i,site in enumerate(['ufpr04','ufpr05','pucpr']) for d in range(20) for j in range(4)]
    split=choose_splits(rows,100)
    assert {k:len(v) for k,v in split.items()}=={'train':70,'val':15,'test':15}
    assert {r['site'] for r in split['test']}=={'pucpr'}
    assert 'pucpr' not in {r['site'] for r in split['train']+split['val']}
    groups=lambda name:{(r['site'],r['day']) for r in split[name]}
    assert not groups('train')&groups('val')
    assert choose_splits(rows,100)==split


def test_streamed_annotations_and_unknown_label_rejection(tmp_path):
    records=[{'id':i,'text':'a'*40000} for i in range(3)]
    path=tmp_path/'samples.json';path.write_text(json.dumps({'samples':records}))
    assert list(iter_samples(path))==records
    sample={'parking_spaces':{'polylines':[{'occupancy_status':'occupied',
            'points':[[[.1,.2],[.3,.2],[.3,.6],[.1,.6]]]}]}}
    line=annotation_lines(sample)[0].split()
    assert line[0]=='1'
    assert list(map(float,line[1:]))==pytest.approx([.2,.4,.2,.4])
    sample['parking_spaces']['polylines'][0]['occupancy_status']='unknown'
    with pytest.raises(ValueError):annotation_lines(sample)


def test_evaluation_counts_false_available_and_missed_bays():
    truth=[{'status':'occupied','box':[0,0,100,100]},
           {'status':'available','box':[110,0,200,100]}]
    learned={'spaces':[{'id':'P01'}],'detections':[{'box':[0,0,100,100]}],
             'occupancy_predictions':{'P01':{'status':'available'}}}
    result=score_image(truth,learned)
    assert result['false_available']==1
    assert result['missed_bays']==1
    assert result['occupied_correct']==0


def test_improvement_gate_rejects_new_false_available_errors():
    baseline={'bay_recall':.9,'bay_precision':.9,'occupied_recall':.9,
              'available_precision':.9,'false_available_rate':0.}
    assert not is_better(baseline,baseline)
    improved=dict(baseline,occupied_recall=.95)
    assert is_better(baseline,improved)
    assert not is_better(baseline,dict(improved,false_available_rate=.01))
    assert not is_better(baseline,dict(improved,bay_recall=.89))
