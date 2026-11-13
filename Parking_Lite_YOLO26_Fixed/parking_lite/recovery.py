"""Recover painted, vehicle-containing bays missed by the learned bay model."""
from copy import deepcopy
import cv2
import numpy as np
from .core import _polygon_px
from .lines import discover_spaces


def ordered_quad(points):
    points=np.asarray(points,np.float32)
    if points.shape!=(4,2) or not cv2.isContourConvex(points):return None
    centre=points.mean(0)
    points=points[np.argsort(np.arctan2(points[:,1]-centre[1],points[:,0]-centre[0]))]
    points=np.roll(points,-int(np.argmin(points.sum(1))),axis=0)
    if np.linalg.norm(points[1]-points[0])>np.linalg.norm(points[2]-points[1]):points=np.roll(points,-1,axis=0)
    return points


def gap_proposals(layout,width,height,verified_anchors=None):
    verified_anchors=verified_anchors or set()
    anchors=[]
    for space in layout['spaces']:
        prediction=layout.get('occupancy_predictions',{}).get(space['id'],{})
        if prediction.get('confidence',0)<.6 and space['id'] not in verified_anchors:continue
        points=ordered_quad(_polygon_px(space['points'],width,height))
        if points is not None:anchors.append(points)
    proposals=[]
    for i,a in enumerate(anchors):
        aw=np.linalg.norm(a[1]-a[0]);ad=np.linalg.norm(a[2]-a[1])
        if min(aw,ad)<4:continue
        u=(a[1]-a[0])/aw;n=np.array([-u[1],u[0]])
        for b in anchors[i+1:]:
            bw=np.linalg.norm(b[1]-b[0]);bd=np.linalg.norm(b[2]-b[1])
            if min(bw,bd)<4 or min(aw,bw)/max(aw,bw)<.7 or min(ad,bd)/max(ad,bd)<.7:continue
            if u@((b[1]-b[0])/bw)<.95:continue
            delta=b.mean(0)-a.mean(0);distance=abs(delta@u)
            if abs(delta@n)>min(ad,bd)*.25:continue
            steps=int(np.rint(distance/((aw+bw)/2)))
            if not 2<=steps<=4 or not .8<distance/steps/((aw+bw)/2)<1.35:continue
            for step in range(1,steps):proposals.append(a+(b-a)*(step/steps))
    return proposals


def paint_support(image,polygon):
    points=ordered_quad(polygon)
    if points is None:return 0.
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);value=hsv[:,:,2]
    white=(hsv[:,:,1]<65)|((hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>80))
    h,w=value.shape;bay_width=np.linalg.norm(points[1]-points[0])
    radius=max(2,min(32,int(bay_width*.16)))
    scores=[]
    for a,b in [(points[1],points[2]),(points[3],points[0])]:
        direction=b-a;length=np.linalg.norm(direction)
        if length<8:return 0.
        normal=np.array([-direction[1],direction[0]])/length
        samples=a+np.linspace(.08,.92,40)[:,None]*direction
        best=0.
        for offset in range(-radius,radius+1):
            def read(delta):
                xy=np.rint(samples+(offset+delta)*normal).astype(int)
                xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
                return value[xy[:,1],xy[:,0]].astype(float),white[xy[:,1],xy[:,0]]
            centre,paint=read(0);left,_=read(max(3,radius));right,_=read(-max(3,radius))
            best=max(best,float(np.mean(paint&(centre>120)&(centre-left>5)&(centre-right>5))))
        scores.append(best)
    return min(scores)


def recover_vehicle_bays(image,layout,vehicles,assignments):
    h,w=image.shape[:2]
    assigned={i for indices in assignments.values() for i in indices}
    unmatched=[(i,v) for i,v in enumerate(vehicles) if i not in assigned]
    if not unmatched:return layout,[]
    # A vehicle alone cannot establish a bay: require observed paint geometry
    # or a bounded gap between two confident, compatible bay anchors.
    verified={sid for sid,indices in assignments.items() if indices}
    proposals=[(points,'painted_gap') for points in gap_proposals(layout,w,h,verified)]
    geometry=discover_spaces(image)
    proposals.extend((_polygon_px(s['points'],w,h).astype(np.float32),'painted_dividers') for s in geometry['spaces'])
    result=deepcopy(layout);added=[]
    for polygon,source in proposals:
        area=abs(cv2.contourArea(polygon));covered=0.
        if area<16:continue
        for space in result['spaces']:
            existing=_polygon_px(space['points'],w,h).astype(np.float32)
            if cv2.isContourConvex(existing):covered+=cv2.intersectConvexConvex(polygon,existing)[0]
        if covered/area>.35:continue
        supported=[]
        for index,vehicle in unmatched:
            if index in assigned:continue
            a,b,c,d=vehicle['box'];centre=((a+c)/2,(b+d)/2)
            if cv2.pointPolygonTest(polygon,centre,False)<0:continue
            rect=np.array([[a,b],[c,b],[c,d],[a,d]],np.float32)
            overlap=cv2.intersectConvexConvex(polygon,rect)[0]
            if overlap/max(1,min(area,(c-a)*(d-b)))>=.35:supported.append(index)
        if not supported:continue
        score=paint_support(image,polygon)
        if score<.25:continue
        sid=f'P{len(result["spaces"])+1:02d}'
        while any(s['id']==sid for s in result['spaces']):sid+='R'
        result['spaces'].append(dict(id=sid,points=(polygon/[w,h]).tolist(),boundary_source=source,
                                     paint_support=round(score,3)))
        result.setdefault('occupancy_predictions',{})[sid]=dict(status='occupied',source='vehicle_painted_bay',
            confidence=max(vehicles[i]['confidence'] for i in supported))
        assigned.update(supported);added.append(sid)
    return result,added
