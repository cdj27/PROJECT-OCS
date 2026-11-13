import cv2
import numpy as np
import pytest
from pathlib import Path
from parking_lite.markings import paint_response
from parking_lite.lines import discover_spaces
from parking_lite.corners import discover_corner_spaces,fuse_corner_boundaries
from parking_lite.orientation import divider_tilt,rotate_canvas
from parking_lite.recovery import paint_support
from tools.augment_markings import marking_variant
from test_corners import bracket_scene

PAINTS=[((240,240,240),70),((0,230,230),70),((230,80,20),70),
        ((20,30,220),70),((40,200,80),70),((35,35,35),200),
        ((145,145,145),85),((90,90,90),30)]


@pytest.mark.parametrize('colour,background',PAINTS)
def test_dividers_support_different_paint_colours_and_light_levels(colour,background):
    image=np.full((500,800,3),background,np.uint8)
    for x in (100,220,340,460,580):cv2.line(image,(x,120),(x,370),colour,6)
    assert len(discover_spaces(image)['spaces'])==4


@pytest.mark.parametrize('colour,background',PAINTS)
def test_four_corner_brackets_support_different_marking_colours(colour,background):
    image,_=bracket_scene();mask=image[:,:,0]>200
    image[:]=background;image[mask]=colour
    assert len(discover_corner_spaces(image)['spaces'])==3


@pytest.mark.parametrize('colour',[(0,230,230),(230,80,20),(20,30,220)])
def test_coloured_diagonal_dividers_retain_their_orientation(colour):
    image=np.full((500,600,3),65,np.uint8)
    for x in (120,220,320,420):cv2.line(image,(x,100),(x,400),colour,5)
    rotated,_=rotate_canvas(image,30)
    assert abs(divider_tilt(rotated)['angle']+30)<3


@pytest.mark.parametrize('colour,background',PAINTS)
def test_recovery_uses_the_same_colour_independent_paint_evidence(colour,background):
    image=np.full((300,300,3),background,np.uint8)
    for x in (100,160):cv2.line(image,(x,50),(x,250),colour,3)
    assert paint_support(image,np.array([[100,50],[160,50],[160,250],[100,250]],np.float32))>=.25


@pytest.mark.parametrize('colour',[(0,230,230),(230,80,20),(40,200,80),(200,200,200)])
def test_uniform_coloured_surfaces_do_not_imply_a_parking_bay(colour):
    image=np.empty((500,800,3),np.uint8);image[:]=colour
    assert not paint_response(image).any()
    assert not discover_spaces(image)['spaces']
    assert not discover_corner_spaces(image)['spaces']


def test_paint_augmentation_is_deterministic_and_preserves_broad_vehicle_content():
    image=np.full((224,224,3),60,np.uint8)
    cv2.line(image,(20,10),(20,210),(240,240,240),3)
    image[60:160,70:155]=(50,80,130)
    first,name=marking_variant(image,'training-photo')
    second,_=marking_variant(image,'training-photo')
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(first[80:140,90:135],image[80:140,90:135])
    assert not np.array_equal(first,image)


def test_mixed_polarity_texture_in_real_photo_does_not_crash_corner_detection():
    image=cv2.imread(str(Path(__file__).resolve().parents[1]/'samples/parking photos/parking.jpg'))
    assert image is not None
    result=discover_corner_spaces(image)
    assert isinstance(result['spaces'],list)


def test_large_corner_combination_cannot_merge_two_confident_known_bays():
    spaces=[dict(id='P01',points=[[.1,.1],[.3,.1],[.3,.8],[.1,.8]]),
            dict(id='P02',points=[[.35,.1],[.55,.1],[.55,.8],[.35,.8]])]
    learned=dict(spaces=spaces,occupancy_predictions={s['id']:dict(status='occupied',confidence=.9) for s in spaces})
    corners=dict(spaces=[dict(id='C01',points=[[.08,.08],[.57,.08],[.57,.82],[.08,.82]])],diagnostics={})
    result=fuse_corner_boundaries(learned,corners,800,500)
    assert len(result['spaces'])==2
    assert result['diagnostics']['rejected_multi_bay_corners']==1
