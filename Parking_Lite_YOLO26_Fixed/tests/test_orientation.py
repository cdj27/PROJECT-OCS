import cv2
import numpy as np
from parking_lite.orientation import rotate_canvas,divider_tilt,centre_tilt,restore_layout
from tools.benchmark_diagonal import polygon_iou,score_polygons
from tools.train_angle_occupancy import angle_variant


def test_rotation_keeps_original_frame_and_inverse_mapping():
    image=np.zeros((300,500,3),np.uint8)
    corners=np.array([[0,0],[500,0],[500,300],[0,300]],np.float32)
    rotated,matrix=rotate_canvas(image,37)
    moved=cv2.transform(corners[None],matrix)[0]
    assert moved.min()>=-1e-4
    assert moved[:,0].max()<=rotated.shape[1]
    assert moved[:,1].max()<=rotated.shape[0]
    restored=cv2.transform(moved[None],cv2.invertAffineTransform(matrix))[0]
    np.testing.assert_allclose(restored,corners,atol=.0001)


def test_restore_produces_diagonal_polygon_instead_of_its_bounding_box():
    image=np.zeros((500,500,3),np.uint8);rotated,matrix=rotate_canvas(image,30)
    points=np.array([[.4,.3],[.6,.3],[.6,.7],[.4,.7]],np.float32)
    learned=dict(spaces=[dict(id='P01',points=points)],occupancy_predictions={'P01':dict(status='available',confidence=.99)})
    result=restore_layout(learned,matrix,image.shape,rotated.shape)
    actual=np.array(result['spaces'][0]['points'])*[500,500]
    expected=cv2.transform((points*[rotated.shape[1],rotated.shape[0]]).astype(np.float32)[None],cv2.invertAffineTransform(matrix))[0]
    assert polygon_iou(actual,expected)>.9999
    assert abs(actual[1,1]-actual[0,1])>20
    assert result['occupancy_predictions']['P01']['status']=='available'


def test_rotation_padding_cannot_invent_a_bay():
    image=np.zeros((500,500,3),np.uint8);rotated,matrix=rotate_canvas(image,45)
    learned=dict(spaces=[dict(id='P01',points=[[0,0],[.1,0],[.1,.1],[0,.1]])],occupancy_predictions={'P01':dict(status='occupied')})
    assert not restore_layout(learned,matrix,image.shape,rotated.shape)['spaces']


def test_painted_divider_tilt_follows_rotation():
    image=np.full((500,600,3),65,np.uint8)
    for x in [120,220,320,420]:cv2.line(image,(x,100),(x,400),(255,255,255),5)
    rotated,_=rotate_canvas(image,30);tilt=divider_tilt(rotated)
    assert abs(tilt['angle']+30)<3
    assert tilt['support']>.25


def test_polygon_metric_rejects_broad_box_around_a_diagonal_bay():
    true=dict(status='occupied',polygon=[[0,0],[20,0],[100,80],[80,80]])
    broad=dict(status='occupied',polygon=[[0,0],[100,0],[100,80],[0,80]])
    assert score_polygons([true],[broad])['matched_bays']==0
    assert score_polygons([true],[true])['correctly_localized_and_classified']==1


def test_repeated_bay_centres_provide_an_independent_tilt_estimate():
    matrix=cv2.getRotationMatrix2D((250,250),30,1);spaces=[]
    for row in range(5):
        for col in range(6):
            centre=cv2.transform(np.array([[[80+60*col,80+70*row]]],np.float32),matrix)[0,0]
            x,y=centre
            spaces.append(dict(points=(np.array([[x-18,y-25],[x+18,y-25],[x+18,y+25],[x-18,y+25]])/500).tolist()))
    estimate=centre_tilt(dict(spaces=spaces),500,500)
    assert abs(estimate['angle']+30)<3
    assert estimate['support']>.35


def test_angle_augmentation_keeps_foreground_and_is_reproducible():
    image=np.zeros((224,224,3),np.uint8);image[:25,:25]=255
    first,angle=angle_variant(image,'held-out-photo')
    second,_=angle_variant(image,'held-out-photo')
    assert np.count_nonzero(first>200)>100
    np.testing.assert_array_equal(first,second)
