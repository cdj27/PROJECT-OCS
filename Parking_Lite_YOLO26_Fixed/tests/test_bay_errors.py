"""Regression coverage for small cars and arrow-marked traffic lanes."""
import cv2
import numpy as np
import pytest
from parking_lite.aisles import road_arrows,exclude_aisle_bays
from parking_lite.verification import verification_crops,restore_crop_vehicle
from parking_lite.engine import ParkingEngine
from parking_lite.slots import discover_parking
from test_slots import Model


def arrow_image(colour=(255,255,255),angle=0):
    im=np.full((400,600,3),65,np.uint8)
    points=np.array([[50,190],[100,190],[100,180],[130,200],[100,220],[100,210],[50,210]])
    cv2.fillPoly(im,[points],colour)
    if angle:
        im=cv2.warpAffine(im,cv2.getRotationMatrix2D((90,200),angle,1),(600,400),borderValue=(65,65,65))
    return im


@pytest.mark.parametrize('colour',[(255,255,255),(0,230,240)])
@pytest.mark.parametrize('angle',[0,30,90])
def test_solid_road_arrow_is_rotation_and_colour_tolerant(colour,angle):
    assert road_arrows(arrow_image(colour,angle))


def test_lines_corners_rectangles_and_triangle_alone_are_not_arrows():
    im=np.full((400,600,3),65,np.uint8)
    cv2.rectangle(im,(30,20),(200,60),(255,255,255),-1)
    cv2.line(im,(240,20),(240,180),(255,255,255),5)
    cv2.line(im,(240,180),(300,180),(255,255,255),5)
    cv2.fillPoly(im,[np.array([[100,250],[70,310],[130,310]])],(255,255,255))
    assert not road_arrows(im)


def test_vehicle_panels_cannot_supply_arrow_evidence():
    assert not road_arrows(arrow_image(),[{'box':[40,170,140,230]}])


def test_aisle_extension_stops_at_transverse_boundary_and_retains_other_row():
    im=arrow_image()
    # A bay end line prevents extrapolation into a real end-on parking bay.
    cv2.line(im,(330,155),(330,245),(255,255,255),5)
    def bay(sid,x,y):return dict(id=sid,points=(np.array([[x,y],[x+70,y],[x+70,y+120],[x,y+120]])/[600,400]).tolist())
    layout={'spaces':[bay('aisle',180,140),bay('real',350,140),bay('other_row',180,270)]}
    result,diagnostic=exclude_aisle_bays(im,layout,[])
    assert [s['id'] for s in result['spaces']]==['real','other_row']
    assert diagnostic['excluded'][0]['id']=='aisle'
    assert len(layout['spaces'])==3


@pytest.mark.parametrize('box',[[25,15,50,50],[25,60,50,95],[30,40,45,65]])
def test_partial_length_car_still_occupies_bay(box):
    spaces=[dict(id='car',points=[[.1,.1],[.6,.1],[.6,.98],[.1,.98]]),
            dict(id='empty',points=[[.65,.1],[.98,.1],[.98,.98],[.65,.98]])]
    hits=ParkingEngine.vehicle_evidence([dict(box=box)],spaces,100,100)
    assert hits=={'car':[0],'empty':[]}


def test_artificial_crop_edges_reject_partial_neighbour_but_image_edges_are_allowed():
    vehicle=dict(label='car',confidence=.9,box=[0,10,40,90])
    assert restore_crop_vehicle(vehicle,(50,50,150,150),200,200) is None
    assert restore_crop_vehicle(vehicle,(0,0,100,100),200,200)['box']==[0,10,40,90]


@pytest.mark.parametrize('quality',['balanced','accurate'])
def test_modes_recheck_empty_predictions_at_close_scale(tmp_path,monkeypatch,quality):
    image=np.zeros((500,800,3),np.uint8)
    learned=discover_parking(Model([(0,.95,[300,200,380,360])]),image)
    engine=ParkingEngine(tmp_path,binary_output=False)
    monkeypatch.setattr(engine,'discover',lambda *args:learned)
    info=engine.model_info();info['small_vehicles']['installed']=True
    monkeypatch.setattr(engine,'model_info',lambda:info)
    monkeypatch.setattr(engine,'load',lambda kind:kind)
    calls=[]
    def detect(model,source,**kwargs):
        calls.append(source.shape)
        if source.shape==image.shape:return dict(vehicles=[],attempts=[])
        # Crop origin (140,80), small car occupies under half of the bay depth.
        return dict(vehicles=[dict(label='car',confidence=.9,box=[175,135,215,195])],attempts=[])
    monkeypatch.setattr('parking_lite.engine.detect_diagnostic',detect)
    report=engine.analyze(image,quality=quality)
    assert report['spaces'][0]['status']=='occupied'
    assert report['diagnostic']['small_vehicle_context_checks']==1
    assert report['vehicles'][0]['detection_view']=='bay_context_crop'


def test_only_unmatched_available_bays_nominate_context_crops():
    spaces=[dict(id='a',points=[[.1,.1],[.2,.1],[.2,.3],[.1,.3]]),
            dict(id='b',points=[[.3,.1],[.4,.1],[.4,.3],[.3,.3]])]
    assert verification_crops(spaces,{'a':{'status':'occupied'},'b':{'status':'available'}},
                              {'a':[],'b':[0]},1000,1000)==[]
