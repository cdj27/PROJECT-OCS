"""Estimate painted-divider tilt and restore aligned detections as polygons."""
import cv2
import numpy as np
from .markings import paint_response


def rotate_canvas(image,angle):
    h,w=image.shape[:2];matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    nw=int(np.ceil(h*abs(matrix[0,1])+w*abs(matrix[0,0])))
    nh=int(np.ceil(h*abs(matrix[0,0])+w*abs(matrix[0,1])))
    matrix[:,2]+=[nw/2-w/2,nh/2-h/2]
    return cv2.warpAffine(image,matrix,(nw,nh),borderValue=(70,70,70)),matrix


def divider_tilt(image):
    scale=min(1.,1000/max(image.shape[:2]));small=cv2.resize(image,None,fx=scale,fy=scale)
    value=paint_response(small);paint=value>8
    lines=cv2.createLineSegmentDetector().detect(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))[0]
    histogram=np.zeros(90);accepted=0
    if lines is None:return dict(angle=0.,support=0.,segments=0)
    h,w=value.shape
    for x1,y1,x2,y2 in lines[:,0]:
        direction=np.array([x2-x1,y2-y1]);length=np.linalg.norm(direction)
        if length<max(15,min(h,w)*.025):continue
        u=direction/length;n=np.array([-u[1],u[0]])
        samples=np.array([x1,y1])+np.linspace(.05,.95,24)[:,None]*direction
        def sample(offset,white=False):
            points=np.rint(samples+offset*n).astype(int)
            points[:,0]=np.clip(points[:,0],0,w-1);points[:,1]=np.clip(points[:,1],0,h-1)
            pixels=value[points[:,1],points[:,0]].astype(float)
            return np.where(paint[points[:,1],points[:,0]],pixels,0) if white else pixels
        centre=np.maximum.reduce([sample(k,True) for k in (-2,0,2)])
        fraction=np.mean((centre>12)&(centre-sample(-7)>7)&(centre-sample(7)>7))
        if fraction<.45:continue
        angle=(np.degrees(np.arctan2(u[1],u[0]))+45)%90-45
        histogram[int(np.rint(angle))%90]+=min(length,100)*fraction
        accepted+=1
    if not accepted:return dict(angle=0.,support=0.,segments=0)
    smoothed=sum(np.roll(histogram,k) for k in range(-3,4))
    peak=int(np.argmax(smoothed));support=float(smoothed[peak]/max(histogram.sum(),1))
    indices=np.array([(peak+k)%90 for k in range(-3,4)])
    angles=((indices-peak+45)%90-45)+peak
    angle=float(np.average(angles,weights=histogram[indices]+1e-9));angle=(angle+45)%90-45
    return dict(angle=round(angle,2),support=round(support,4),segments=accepted)


def restore_layout(learned,matrix,original_shape,rotated_shape):
    h,w=original_shape[:2];rh,rw=rotated_shape[:2]
    inverse=cv2.invertAffineTransform(matrix);frame=np.array([[0,0],[w,0],[w,h],[0,h]],np.float32)
    spaces=[];predictions={};detections=[]
    for space in learned['spaces']:
        points=np.array(space['points'],np.float32)*[rw,rh]
        polygon=cv2.transform(points.astype(np.float32)[None],inverse)[0]
        area=abs(cv2.contourArea(polygon));intersection,clipped=cv2.intersectConvexConvex(polygon,frame)
        if area<16 or clipped is None or intersection/area<.75:continue
        points=clipped.reshape(-1,2)/[w,h]
        sid=f'P{len(spaces)+1:02d}'
        spaces.append(dict(id=sid,points=points.tolist(),boundary_source='rotation_aligned'))
        predictions[sid]=dict(learned['occupancy_predictions'][space['id']])
        lo=clipped.reshape(-1,2).min(0);hi=clipped.reshape(-1,2).max(0)
        detections.append(dict(box=[*lo.tolist(),*hi.tolist()]))
    return dict(version=1,source='rotation_aligned',spaces=spaces,occupancy_predictions=predictions,
                detections=detections,diagnostics=dict(learned.get('diagnostics',{})),review_required=True)


def centre_tilt(learned,width,height):
    """A second orientation estimate from repeated bay-centre spacing."""
    points=[np.array(s['points'])*[width,height] for s in learned['spaces']]
    if len(points)<8:return dict(angle=0.,support=0.)
    centres=np.array([p.mean(0) for p in points]);sizes=np.array([np.ptp(p,axis=0).min() for p in points])
    typical=max(4,float(np.median(sizes)));histogram=np.zeros(90)
    for i,c in enumerate(centres):
        delta=centres-c;distances=np.linalg.norm(delta,axis=1)
        nearest=np.argsort(distances)[1:5]
        for j in nearest:
            if not typical*.4<distances[j]<typical*2.5:continue
            angle=(np.degrees(np.arctan2(delta[j,1],delta[j,0]))+45)%90-45
            histogram[int(np.rint(angle))%90]+=min(2.,typical/distances[j])
    if not histogram.sum():return dict(angle=0.,support=0.)
    smoothed=sum(np.roll(histogram,k) for k in range(-3,4));peak=int(smoothed.argmax())
    ids=np.array([(peak+k)%90 for k in range(-3,4)])
    angles=((ids-peak+45)%90-45)+peak
    # Centre vectors use the same image-coordinate sign as painted segments.
    angle=float(np.average(angles,weights=histogram[ids]+1e-9));angle=(angle+45)%90-45
    return dict(angle=round(angle,2),support=float(smoothed[peak]/histogram.sum()))
