import numpy as np
from pathlib import Path
from parking_lite.core import save_layout,load_layout,analyze_image,detect_diagnostic

def test_absent_layout():
    report=analyze_image(np.zeros((100,100,3),dtype=np.uint8),[],None)
    assert report['mapped'] is False and report['counts']['total']==0

def test_no_detection_means_unknown(tmp_path):
    p=tmp_path/'a.json'; save_layout(p,[[(0,0),(50,0),(50,50),(0,50)]],100,100)
    result=analyze_image(np.zeros((100,100,3),dtype=np.uint8),[],load_layout(p))
    assert result['counts']=={'occupied':0,'available':0,'unknown':1,'total':1}

def test_detected_vehicles_and_override(tmp_path):
    p=tmp_path/'layout.json';save_layout(p,[[(0,0),(49,0),(49,100),(0,100)],[(51,0),(100,0),(100,100),(51,100)]],100,100)
    veh=[{'box':[10,10,30,50],'label':'car','confidence':.9}]
    r=analyze_image(np.zeros((100,100,3),dtype=np.uint8),veh,load_layout(p))
    assert r['counts']=={'occupied':1,'available':0,'unknown':1,'total':2}
    m=analyze_image(np.zeros((100,100,3),dtype=np.uint8),veh,load_layout(p),{'P02':'occupied'})
    assert m['counts']['occupied']==2 and m['spaces'][1]['source']=='manual'

class Box:
    def __init__(self,label_id):
        self.cls=Arr([label_id]);self.conf=Arr([.12]);self.xyxy=[Arr([1,2,20,30])]
class Arr:
    def __init__(self,v):self.v=v
    def item(self):return self.v[0]
    def tolist(self):return self.v
class Result:
    def __init__(self,label_id):self.boxes=[Box(label_id)];self.names={0:'person',2:'car'}
class FakeModel:
    def __init__(self):self.calls=0
    def predict(self,**kwargs):
        self.calls+=1
        return [Result(0 if self.calls==1 else 2)]

def test_retry_on_nonvehicle():
    model=FakeModel()
    out=detect_diagnostic(model,np.zeros((100,100,3),dtype=np.uint8))
    assert model.calls==2 and len(out['vehicles'])==1 and out['vehicles'][0]['label']=='car'
    assert out['attempts'][0]['raw_count']==1
