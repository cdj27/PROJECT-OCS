import json
from pathlib import Path
import numpy as np
import pytest
from parking_lite.engine import ParkingEngine
from parking_lite.session import ImageStore
from test_batch import photo
from test_slots import Model
from parking_lite.slots import discover_parking


def test_photo_ownership_and_immutable_results(tmp_path):
    store=ImageStore(tmp_path)
    first=store.import_photo(photo(),'one.png')
    _,image,folder=store.snapshot(first['image_id'])
    result=store.save_result(first['image_id'],{'counts':{'total':0}},image)
    second=store.import_photo(photo(90),'two.png')
    assert first['image_id']!=second['image_id']
    assert not (store.folder(second['image_id'])/'layout.json').exists()
    with pytest.raises(ValueError):store.snapshot(first['image_id'])
    assert json.loads(store.result_file(first['image_id'],result['result_id'],'analysis.json').read_text())['counts']=={'total':0}
    with pytest.raises(ValueError):store.folder('../outside')


def test_vehicle_near_adjacent_empty_bay_prevents_false_available(tmp_path,monkeypatch):
    image=np.zeros((100,200,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[10,10,110,90]),(1,.95,[90,10,190,90])]),image)
    engine=ParkingEngine(tmp_path,binary_output=False)
    monkeypatch.setattr(engine,'discover',lambda *args:learned)
    monkeypatch.setattr(engine,'load',lambda *args:object())
    vehicles=[{'label':'car','confidence':.9,'box':[95,20,175,80]}]
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',lambda *args,**kwargs:{'vehicles':list(vehicles),'attempts':[]})
    result=engine.analyze(image)
    assert result['spaces'][0]['status']=='unknown'
    assert result['spaces'][0]['source']=='vehicle_boundary_overlap'
    assert result['spaces'][1]['status']=='occupied'
    assert result['spaces'][1]['vehicle_detected']


def test_clipped_thin_detections_are_not_enumerated_as_whole_bays():
    image=np.zeros((100,200,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[0,92,100,100]),(1,.95,[110,10,190,90])]),image)
    assert len(learned['spaces'])==1


@pytest.mark.parametrize('probability,car_left,expected',[(.999,95,'available'),(.98,95,'unknown'),(.999,75,'unknown')])
def test_neighbour_overlap_requires_strong_independent_empty_evidence(tmp_path,monkeypatch,probability,car_left,expected):
    image=np.zeros((100,200,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[10,10,110,90]),(1,.95,[90,10,190,90])]),image)
    engine=ParkingEngine(tmp_path,verify_boundary_overlap=True,binary_output=False)
    monkeypatch.setattr(engine,'discover',lambda *args:learned)
    original_info=engine.model_info()
    original_info['occupancy']['installed']=True
    monkeypatch.setattr(engine,'model_info',lambda:original_info)
    class Classifier:
        def classify(self,*args):
            return dict(status='available',available_probability=probability,confidence=probability)
    monkeypatch.setattr(engine,'load',lambda kind:Classifier() if kind=='occupancy' else object())
    vehicles=[{'label':'car','confidence':.9,'box':[car_left,20,car_left+80,80]}]
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',lambda *args,**kwargs:{'vehicles':vehicles,'attempts':[]})
    result=engine.analyze(image)
    assert result['spaces'][0]['status']==expected
    assert result['spaces'][1]['status']=='occupied'


def test_extra_scale_keeps_baseline_bays_and_flags_conflicts():
    image=np.zeros((100,300,3),np.uint8)
    baseline=discover_parking(Model([(0,.95,[10,10,90,90]),(1,.95,[110,10,190,90])]),image)
    detail=discover_parking(Model([(1,.95,[10,10,90,90])]),image)
    merged=ParkingEngine.merge_scales(baseline,detail)
    assert len(merged['spaces'])==2
    assert merged['occupancy_predictions']['P01']['status']=='unknown'
    assert merged['occupancy_predictions']['P02']['status']=='occupied'
    assert baseline['occupancy_predictions']['P01']['status']=='available'


def test_accurate_mode_checks_empty_bay_crop(tmp_path,monkeypatch):
    image=np.zeros((100,100,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[10,10,90,90])]),image)
    engine=ParkingEngine(tmp_path,binary_output=False);calls=[]
    monkeypatch.setattr(engine,'discover',lambda *args:learned)
    monkeypatch.setattr(engine,'load',lambda *args:object())
    def detect(model,source,**options):
        calls.append(source.shape)
        return {'vehicles':[] if len(calls)==1 else [{'label':'car','confidence':.9,'box':[20,20,60,60]}],'attempts':[]}
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',detect)
    report=engine.analyze(image,quality='accurate')
    assert len(calls)==2
    assert report['diagnostic']['verified_empty_candidates']==1
    assert report['spaces'][0]['status']=='occupied'


def test_corrections_do_not_rerun_models_or_change_other_photos(tmp_path,monkeypatch):
    import web_app
    store=ImageStore(tmp_path);item=store.import_photo(photo(),'test.png')
    monkeypatch.setattr(web_app,'images',store)
    class FakePipeline:
        calls=0
        def analyze(self,*args,**kwargs):
            self.calls+=1
            return {'vehicles':[],'spaces':[{'id':'P01','status':'occupied','source':'parking_detector',
              'automatic_status':'occupied','automatic_source':'parking_detector','automatic_reason':'Model evidence',
              'polygon':[[10,10],[100,10],[100,180],[10,180]],'vehicle_detected':False}],
              'counts':{'total':1,'occupied':1,'available':0,'unknown':0},'diagnostic':{}}
    pipeline=FakePipeline();monkeypatch.setattr(web_app,'pipeline',pipeline)
    original=web_app.analyze(web_app.AnalyzeBody(image_id=item['image_id']))
    corrected=web_app.correct_result(item['image_id'],original['result_id'],web_app.CorrectionBody(statuses={'P01':'available'}))
    assert corrected['spaces'][0]['status']=='available'
    assert corrected['spaces'][0]['source']=='manual'
    assert pipeline.calls==1
    restored=web_app.correct_result(item['image_id'],corrected['result_id'],web_app.CorrectionBody(statuses={'P01':''}))
    assert restored['spaces'][0]['status']=='occupied'
    store.import_photo(photo(90),'other.png')
    with pytest.raises(Exception):web_app.analyze(web_app.AnalyzeBody(image_id=item['image_id']))
    assert json.loads(store.result_file(item['image_id'],original['result_id'],'analysis.json').read_text())['spaces'][0]['status']=='occupied'
