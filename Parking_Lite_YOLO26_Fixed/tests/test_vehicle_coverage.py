import cv2
import numpy as np
import pytest
from parking_lite.core import deduplicate_vehicles,_parse_predictions
from parking_lite.engine import ParkingEngine
from parking_lite.recovery import recover_vehicle_bays
from parking_lite.orientation import rotate_canvas


def scene(paint=True):
    image=np.full((300,300,3),60,np.uint8)
    if paint:
        for x in (100,160):cv2.line(image,(x,50),(x,250),(255,255,255),3)
    spaces=[dict(id='P01',points=(np.array([[30,50],[90,50],[90,250],[30,250]])/300).tolist()),
            dict(id='P02',points=(np.array([[170,50],[230,50],[230,250],[170,250]])/300).tolist())]
    return image,dict(spaces=spaces,occupancy_predictions={s['id']:dict(status='occupied',confidence=.95) for s in spaces})


@pytest.mark.parametrize('label',['motorcycle','van','truck','bus','bicycle','car'])
def test_vehicle_recovers_a_missing_painted_bay(label):
    image,layout=scene();vehicles=[dict(label=label,confidence=.9,box=[115,90,145,155])]
    recovered,ids=recover_vehicle_bays(image,layout,vehicles,dict(P01=[],P02=[]))
    assert ids==['P03']
    hits=ParkingEngine.vehicle_evidence(vehicles,recovered['spaces'],300,300)
    assert hits['P03']==[0]
    assert recovered['occupancy_predictions']['P03']['status']=='occupied'
    assert len(layout['spaces'])==2


def test_no_paint_cannot_invent_a_bay_around_a_vehicle():
    image,layout=scene(False)
    _,ids=recover_vehicle_bays(image,layout,[dict(label='motorcycle',confidence=.9,box=[115,90,145,155])],dict(P01=[],P02=[]))
    assert ids==[]


def test_vehicle_evidence_can_corroborate_weak_bay_anchor_scores():
    image,layout=scene()
    for prediction in layout['occupancy_predictions'].values():prediction['confidence']=.3
    vehicles=[dict(label='car',confidence=.8,box=[40,80,80,200]),
              dict(label='car',confidence=.4,box=[180,80,220,200]),
              dict(label='motorcycle',confidence=.6,box=[115,90,145,155])]
    assignments=ParkingEngine.vehicle_evidence(vehicles,layout['spaces'],300,300)
    result,ids=recover_vehicle_bays(image,layout,vehicles,assignments)
    assert ids==['P03']


def test_paint_matching_tolerance_scales_with_the_bay_width():
    image,layout=scene();image=cv2.resize(image,(1200,1200),interpolation=cv2.INTER_NEAREST)
    for space in layout['spaces']:
        points=np.array(space['points']);points[:,0]+=.02;space['points']=points.tolist()
    vehicles=[dict(label='motorcycle',confidence=.9,box=[460,360,580,620])]
    _,ids=recover_vehicle_bays(image,layout,vehicles,dict(P01=[],P02=[]))
    assert ids==['P03']


def test_road_vehicle_outside_the_anchored_row_does_not_add_a_bay():
    image,layout=scene()
    _,ids=recover_vehicle_bays(image,layout,[dict(label='truck',confidence=.9,box=[260,90,290,155])],dict(P01=[],P02=[]))
    assert ids==[]


def test_a_vehicle_already_in_a_known_bay_does_not_add_another():
    image,layout=scene()
    _,ids=recover_vehicle_bays(image,layout,[dict(label='motorcycle',confidence=.9,box=[40,90,70,155])],dict(P01=[0],P02=[]))
    assert ids==[]


def test_painted_gap_recovery_also_works_for_diagonal_bays():
    image,layout=scene();rotated,matrix=rotate_canvas(image,30);h,w=rotated.shape[:2]
    for space in layout['spaces']:
        points=np.array(space['points'],np.float32)*300
        space['points']=(cv2.transform(points[None],matrix)[0]/[w,h]).tolist()
    centre=cv2.transform(np.array([[[130,120]]],np.float32),matrix)[0,0]
    x,y=centre;vehicles=[dict(label='motorcycle',confidence=.9,box=[x-10,y-20,x+10,y+20])]
    result,ids=recover_vehicle_bays(rotated,layout,vehicles,dict(P01=[],P02=[]))
    assert ids==['P03']
    points=np.array(result['spaces'][-1]['points'])*[w,h]
    assert abs(points[1,1]-points[0,1])>10


def test_nested_duplicate_motorcycle_boxes_are_counted_once():
    vehicles=[dict(label='motorcycle',confidence=.9,box=[10,10,40,100]),
              dict(label='motorcycle',confidence=.5,box=[15,50,35,95]),
              dict(label='motorcycle',confidence=.8,box=[45,10,75,100])]
    assert len(deduplicate_vehicles(vehicles))==2


@pytest.mark.parametrize('label,expected',[('Motorbike','motorcycle'),('lorry','truck'),('van','van'),('bicycle','bicycle'),('person',None)])
def test_vehicle_class_filter_supports_aliases_and_excludes_people(label,expected):
    import torch
    class Box:
        cls=torch.tensor(0);conf=torch.tensor(.9);xyxy=torch.tensor([[1,2,30,40]])
    class Result:
        names={0:label};boxes=[Box()]
    parsed=_parse_predictions(Result())
    assert ([p['label'] for p in parsed])==([] if expected is None else [expected])


def test_long_vehicle_can_occupy_two_bays():
    image,layout=scene()
    vehicles=[dict(label='truck',confidence=.9,box=[25,40,235,260])]
    assigned=ParkingEngine.vehicle_evidence(vehicles,layout['spaces'],300,300)
    assert assigned==dict(P01=[0],P02=[0])
