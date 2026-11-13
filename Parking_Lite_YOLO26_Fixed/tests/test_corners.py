import cv2
import numpy as np
from parking_lite.corners import discover_corner_spaces,fuse_corner_boundaries


def bracket_scene(count=3,missing=None):
    image=np.full((700,1100,3),55,np.uint8);polygons=[]
    for bay in range(count):
        x=110+bay*270;y=140;w=160;h=330;leg=25
        corners=[(x,y,1,1),(x+w,y,-1,1),(x+w,y+h,-1,-1),(x,y+h,1,-1)]
        polygons.append([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        for index,(a,b,sx,sy) in enumerate(corners):
            if missing==(bay,index):continue
            cv2.line(image,(a,b),(a+sx*leg,b),(245,245,245),5)
            cv2.line(image,(a,b),(a,b+sy*leg),(245,245,245),5)
    return image,polygons


def test_four_brackets_discover_separate_bays_without_lines():
    image,truth=bracket_scene()
    result=discover_corner_spaces(image)
    assert len(result['spaces'])==3,result['diagnostics']
    for detected,expected in zip(result['spaces'],truth):
        points=np.array(detected['points'])*[1100,700]
        assert np.max(np.linalg.norm(points-np.array(expected),axis=1))<8
        assert detected['boundary_source']=='corner_brackets'
    assert not result['occupancy_predictions']


def test_rotated_and_moderate_perspective_brackets():
    image,_=bracket_scene()
    assert len(discover_corner_spaces(cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))['spaces'])==3
    matrix=cv2.getRotationMatrix2D((550,350),12,1.)
    rotated=cv2.warpAffine(image,matrix,(1100,700),borderValue=(55,55,55))
    assert len(discover_corner_spaces(rotated)['spaces'])==3
    source=np.float32([[0,0],[1099,0],[1099,699],[0,699]])
    target=np.float32([[60,30],[1020,0],[1090,660],[0,690]])
    perspective=cv2.warpPerspective(image,cv2.getPerspectiveTransform(source,target),(1100,700),borderValue=(55,55,55))
    assert len(discover_corner_spaces(perspective)['spaces'])==3


def test_missing_corner_does_not_invent_a_bay():
    image,_=bracket_scene(count=1,missing=(0,2))
    assert not discover_corner_spaces(image)['spaces']


def test_blank_and_white_rectangles_are_not_corner_brackets():
    image=np.full((700,1100,3),55,np.uint8)
    assert not discover_corner_spaces(image)['spaces']
    for x in (110,380,650):cv2.rectangle(image,(x,140),(x+160,470),(245,245,245),5)
    assert not discover_corner_spaces(image)['spaces']


def test_compressed_blurred_and_dim_paint():
    image,_=bracket_scene()
    image=np.where(image>200,155,image).astype(np.uint8)
    image=cv2.GaussianBlur(image,(3,3),.6)
    encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])[1]
    compressed=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    assert len(discover_corner_spaces(compressed)['spaces'])==3


def test_corner_refinement_keeps_occupancy_and_avoids_duplicate_bays():
    image,_=bracket_scene(count=1)
    corners=discover_corner_spaces(image)
    learned={'version':1,'spaces':[{'id':'P01','points':[[.09,.18],[.26,.18],[.26,.7],[.09,.7]]}],
             'occupancy_predictions':{'P01':{'status':'occupied','confidence':.9,'source':'parking_detector'}},'diagnostics':{}}
    merged=fuse_corner_boundaries(learned,corners,1100,700)
    assert len(merged['spaces'])==1
    assert merged['spaces'][0]['boundary_source']=='corner_brackets'
    assert merged['occupancy_predictions']['P01']['status']=='occupied'


def test_one_broad_model_box_cannot_collapse_two_corner_bays():
    image,_=bracket_scene(count=2)
    corners=discover_corner_spaces(image)
    learned={'spaces':[{'id':'P01','points':[[.1,.2],[.5,.2],[.5,.68],[.1,.68]]}],
             'occupancy_predictions':{'P01':{'status':'available','confidence':.95,'source':'parking_detector'}},'diagnostics':{}}
    merged=fuse_corner_boundaries(learned,corners,1100,700)
    assert len(merged['spaces'])==2
    assert len({s['id'] for s in merged['spaces']})==2
    assert all(p['status']=='unknown' for p in merged['occupancy_predictions'].values())


def test_corner_discovery_does_not_assume_empty_when_models_fail(tmp_path,monkeypatch):
    from parking_lite.engine import ParkingEngine
    image,_=bracket_scene()
    engine=ParkingEngine(tmp_path,binary_output=False)
    monkeypatch.setattr(engine,'load',lambda *args:object())
    monkeypatch.setattr('parking_lite.engine.discover_parking',lambda *args,**kwargs:{'spaces':[],'occupancy_predictions':{},'diagnostics':{}})
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',lambda *args,**kwargs:{'vehicles':[],'attempts':[]})
    report=engine.analyze(image)
    assert report['counts']=={'total':3,'occupied':0,'available':0,'unknown':3}
    assert all(s['boundary_source']=='corner_brackets' for s in report['spaces'])


def test_corner_bays_get_separate_occupancy_evidence(tmp_path,monkeypatch):
    from parking_lite.engine import ParkingEngine
    image,_=bracket_scene();engine=ParkingEngine(tmp_path,binary_output=False)
    monkeypatch.setattr(engine,'load',lambda *args:object())
    def parking(model,source,**kwargs):
        if source.shape==image.shape:return {'spaces':[],'occupancy_predictions':{},'diagnostics':{}}
        return {'spaces':[{'id':'P01','points':[[.15,.08],[.85,.08],[.85,.92],[.15,.92]]}],
                'occupancy_predictions':{'P01':{'status':'available','confidence':.95,'source':'parking_detector'}}}
    def cars(model,source,**kwargs):
        return {'vehicles':[{'label':'car','box':[140,190,240,420],'confidence':.9}] if source.shape==image.shape else [],'attempts':[]}
    monkeypatch.setattr('parking_lite.engine.discover_parking',parking)
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',cars)
    report=engine.analyze(image,engine='corner_brackets')
    assert report['counts']=={'total':3,'occupied':1,'available':2,'unknown':0}
    assert report['spaces'][0]['vehicle_detected']
    assert report['diagnostic']['corner_crop_checks']==2


def test_empty_classifier_cannot_override_conflicting_occupied_model(tmp_path,monkeypatch):
    from parking_lite.engine import ParkingEngine
    image=np.zeros((200,200,3),np.uint8)
    learned={'source':'corner_brackets','spaces':[{'id':'P01','points':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'boundary_source':'corner_brackets'}],
             'occupancy_predictions':{'P01':{'status':'occupied','confidence':.95,'source':'parking_detector'}},'diagnostics':{}}
    engine=ParkingEngine(tmp_path,binary_output=False)
    class Classifier:
        def classify(self,*args):return {'status':'available','confidence':.99,'source':'bay_classifier'}
    monkeypatch.setattr(engine,'discover',lambda *args:learned)
    monkeypatch.setattr(engine,'model_info',lambda:{'occupancy':{'installed':True}})
    monkeypatch.setattr(engine,'load',lambda kind:Classifier() if kind=='occupancy' else object())
    monkeypatch.setattr('parking_lite.engine.discover_parking',lambda *args,**kwargs:{
        'spaces':[{'id':'P01','points':[[0,0],[1,0],[1,1],[0,1]]}],
        'occupancy_predictions':{'P01':{'status':'available','confidence':.99,'source':'parking_detector'}}})
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',lambda *args,**kwargs:{'vehicles':[],'attempts':[]})
    assert engine.analyze(image)['spaces'][0]['status']=='unknown'
