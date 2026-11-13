import cv2
import numpy as np
from parking_lite.lines import discover_spaces
from parking_lite.core import detect_diagnostic, classify_spaces, analyze_image
from test_corrected import Arr, Box


def test_white_line_row_and_rotation():
    image=np.full((500,800,3),70,np.uint8)
    for x in (100,220,340,460,580):
        cv2.line(image,(x,120),(x,370),(240,240,240),6)
    for frame in (image,cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)):
        layout=discover_spaces(frame)
        assert len(layout['spaces'])==4
        assert layout['review_required']
        assert all(0<=c<=1 for s in layout['spaces'] for p in s['points'] for c in p)


def test_blank_has_no_bays_and_yellow_markings_are_supported():
    image=np.full((500,800,3),70,np.uint8)
    assert not discover_spaces(image)['spaces']
    for x in (100,220,340,460,580):
        cv2.line(image,(x,120),(x,370),(0,230,230),6)
    assert len(discover_spaces(image)['spaces'])==4


def test_tiled_coordinates_and_duplicates():
    class Result:
        names={2:'car'}
        def __init__(self,box):
            b=Box(2);b.xyxy=[Arr(box)];self.boxes=[b]
    class Model:
        def __init__(self):self.n=0
        def predict(self,**kwargs):
            self.n+=1
            # Full frame and first crop identify the same car.
            return [Result([10,10,30,30] if self.n<=2 else [40,40,55,55])]
    report=detect_diagnostic(Model(),np.zeros((100,100,3),np.uint8),tiled=True)
    boxes=[v['box'] for v in report['vehicles']]
    assert boxes.count([10,10,30,30])==1
    assert [75,75,90,90] in boxes
    assert len(report['attempts'])==5


def test_trained_bay_classifier_can_identify_empty_scene():
    class Probs:
        top1=0;top1conf=Arr([.95])
    class Result:
        probs=Probs();names={0:'available',1:'occupied'}
    class Model:
        def predict(self,**kwargs):
            assert kwargs['source'].shape==(224,224,3)
            return [Result()]
    image=np.zeros((100,100,3),np.uint8)
    layout={'spaces':[{'id':'P01','points':[[0,0],[1,0],[1,1],[0,1]]}]}
    predictions=classify_spaces(Model(),image,layout)
    result=analyze_image(image,[],layout,occupancy_predictions=predictions)
    assert result['spaces'][0]['status']=='available'
    assert result['spaces'][0]['source']=='occupancy_classifier'
    Result.probs.top1conf=Arr([.55])
    assert classify_spaces(Model(),image,layout)['P01']['status']=='unknown'
