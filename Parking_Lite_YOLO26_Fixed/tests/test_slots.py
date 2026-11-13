import numpy as np
from parking_lite.slots import discover_parking,match_occupancy
from parking_lite.core import analyze_image
from test_corrected import Arr


class Model:
    names={0:'space-empty',1:'space-occupied'}
    def __init__(self,rows):self.rows=rows
    def predict(self,**kwargs):
        class Result:pass
        result=Result();result.names=self.names;result.boxes=[]
        for label,score,box in self.rows:
            class Box:pass
            b=Box();b.cls=Arr([label]);b.conf=Arr([score]);b.xyxy=[Arr(box)]
            result.boxes.append(b)
        return [result]


def test_learned_bays_and_confidence():
    image=np.zeros((100,300,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[0,0,90,100]),(1,.9,[100,0,190,100]),(0,.6,[200,0,290,100])]),image)
    r=analyze_image(image,[],learned,occupancy_predictions=learned['occupancy_predictions'])
    assert r['counts']=={'total':3,'available':1,'occupied':1,'unknown':1}
    assert r['spaces'][0]['source']=='parking_detector'


def test_conflicting_predictions_are_unknown():
    image=np.zeros((100,100,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[0,0,90,90]),(1,.7,[1,1,89,89])]),image)
    assert len(learned['spaces'])==1
    assert learned['occupancy_predictions']['P01']['status']=='unknown'


def test_matching_edited_bay_ids_and_vehicle_override():
    image=np.zeros((100,200,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[0,0,90,100])]),image)
    layout={'spaces':[{'id':'edited','points':[[0,0],[.45,0],[.45,1],[0,1]]},
                      {'id':'unmatched','points':[[.5,0],[1,0],[1,1],[.5,1]]}]}
    predictions=match_occupancy(layout,learned,200,100)
    assert predictions['edited']['status']=='available'
    assert 'unmatched' not in predictions
    vehicle={'label':'car','confidence':.9,'box':[10,10,70,90]}
    report=analyze_image(image,[vehicle],layout,occupancy_predictions=predictions)
    assert report['spaces'][0]['status']=='occupied'
    assert report['spaces'][1]['status']=='unknown'


def test_one_learned_box_cannot_mark_two_edited_bays_available():
    image=np.zeros((100,100,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[0,0,100,100])]),image)
    layout={'spaces':[{'id':key,'points':poly} for key,poly in
              [('a',[[0,0],[.5,0],[.5,1],[0,1]]),('b',[[.5,0],[1,0],[1,1],[.5,1]])]]}
    assert len(match_occupancy(layout,learned,100,100))==1
