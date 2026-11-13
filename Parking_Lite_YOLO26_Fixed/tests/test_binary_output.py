import numpy as np
import pytest
import torch
from parking_lite.engine import ParkingEngine
from tools.train_multidataset import cnr_split,extended_split,metrics


@pytest.mark.parametrize('probability,expected',[(.51,'available'),(.49,'occupied')])
def test_binary_resolution_uses_content_even_below_abstention_threshold(tmp_path,monkeypatch,probability,expected):
    engine=ParkingEngine(tmp_path)
    info=engine.model_info();info['occupancy']['installed']=True
    monkeypatch.setattr(engine,'model_info',lambda:info)
    class Classifier:
        def classify(self,*args):return dict(status='unknown',available_probability=probability)
    monkeypatch.setattr(engine,'load',lambda *args:Classifier())
    space=dict(status='unknown',source='parking_detector',polygon=[[1,1],[19,1],[19,19],[1,19]])
    assert engine.resolve_binary_status(space,np.zeros((20,20,3),np.uint8))
    assert space['status']==expected
    assert space['source']=='binary_bay_classifier'


def test_binary_resolution_does_not_override_vehicle_evidence(tmp_path):
    space=dict(status='occupied',source='vehicle_detector')
    assert not ParkingEngine(tmp_path).resolve_binary_status(space,None)
    assert space['status']=='occupied'


def test_missing_model_does_not_declare_empty(tmp_path):
    space=dict(status='unknown',source='parking_detector',polygon=[[1,1],[19,1],[19,19],[1,19]])
    ParkingEngine(tmp_path).resolve_binary_status(space,np.zeros((20,20,3),np.uint8))
    assert space['status']=='occupied'


def test_full_pipeline_counts_have_only_binary_statuses(tmp_path,monkeypatch):
    engine=ParkingEngine(tmp_path)
    layout=dict(spaces=[dict(id='P01',points=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]])])
    monkeypatch.setattr(engine,'discover',lambda *args:layout)
    monkeypatch.setattr(engine,'load',lambda *args:object())
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',lambda *args,**kwargs:dict(vehicles=[],attempts=[]))
    image=np.zeros((100,100,3),np.uint8)
    report=engine.analyze(image)
    assert report['counts']==dict(occupied=1,available=0,total=1)
    assert report['spaces'][0]['automatic_status']=='occupied'
    with pytest.raises(ValueError,match='Invalid manual'):
        engine.analyze(image,manual_statuses={'P01':'unknown'})


def test_cnr_split_keeps_camera_b_and_entire_frames_out_of_training():
    names=[f'A/{label}/20150703_{i:04d}_{slot}.jpg' for i in range(100) for slot,label in [(1,'busy'),(2,'free')]]
    names+=['B/free/20150708_1000_1.jpg']
    split=cnr_split(names)
    assert split['B/free/20150708_1000_1.jpg']=='cnr_test'
    assert len([s for s in split.values() if s=='train'])==150
    assert 'A/busy/20150703_0075_1.jpg' not in split
    grouped={}
    for name,value in split.items():
        frame=name.split('/')[0]+'/'+name.rsplit('/',1)[1].rsplit('_',1)[0]
        grouped.setdefault(frame,set()).add(value)
    assert all(len(values)==1 for values in grouped.values())


def test_binary_metrics_count_low_score_errors():
    result=metrics(torch.tensor([[.51,.49],[.9,.1]]),torch.tensor([1,0]))
    assert result['accuracy']==.5
    assert result['false_available']==1


def test_extended_split_separates_both_cameras_and_days():
    records=[(f'SUNNY/2015-12-{day:02d}/camera{cam}/S_{day}_{cam}.jpg','0') for day in range(1,24) for cam in range(1,10)]
    split=extended_split(records)
    dates={k:set() for k in ('train','ext_val','ext_test')}
    for name,(key,_) in split.items():
        date=name.split('/')[1];camera=int(name.split('/')[2][6:]);dates[key].add(date)
        assert (camera<=6 if key=='train' else camera==7 if key=='ext_val' else camera in (8,9))
    assert all(dates.values())
    assert not dates['train']&dates['ext_val']
    assert not dates['train']&dates['ext_test']
    assert not dates['ext_val']&dates['ext_test']
